#!/usr/bin/env python3
"""
RVQ with Broad Phonetic Classes (Loweimi et al., IEEE/ACM TASLP 2023; see
scripts/broad_phonetic_classes.py).

The main objective is the broad phonetic class (with --bpc_type 8class the 8
classes Affricate, Diphthong, Fricative, Nasal, Plosive, Semi-vowel, Vowel,
Silence); the 43-class phoneme loss is an auxiliary term weighted by
--fine_weight.

Rationale: with 128 codes and 8 target classes, each VQ code can specialise in
one broad phonetic class.

Data pipeline: identical to train_rvq_supervised.py.
Only the classification target changes.
"""
import os
import re
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
import logging
from pathlib import Path
import sys
import argparse

from paths import ALIGNMENTS_DIR, CHECKPOINTS_DIR, MANIFEST_TRAIN_CLEAN_100, MANIFEST_DEV_CLEAN

from data.phoneme_vocab import PHONEME_TO_ID, normalize_phoneme, VOCAB_SIZE, ID_TO_PHONEME
from data.dataset import reanchor_audio_path
from scripts.broad_phonetic_classes import (
    build_phoneme_to_bpc_table,
    IDX_TO_BPC_8, IDX_TO_BPC_3_CVS, IDX_TO_BPC_3_VUS,
    BPC_8_TO_IDX, BPC_3_CVS_TO_IDX, BPC_3_VUS_TO_IDX,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# === DATA PIPELINE ===

ALIGNMENT_BASE = str(ALIGNMENTS_DIR)


def parse_textgrid(textgrid_path):
    """Parse TextGrid file to extract phone intervals."""
    intervals = []
    with open(textgrid_path, 'r') as f:
        content = f.read()

    phone_match = re.search(r'name = "phones".*?intervals: size = (\d+)(.*?)(?:item \[|$)',
                           content, re.DOTALL)
    if not phone_match:
        return intervals

    intervals_text = phone_match.group(2)
    pattern = r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'

    for match in re.finditer(pattern, intervals_text):
        xmin, xmax, phone = float(match.group(1)), float(match.group(2)), match.group(3)
        if phone and phone not in ('', 'sp', 'spn', 'sil', ''):
            intervals.append((xmin, xmax, phone))

    return intervals


def get_frame_labels(intervals, num_frames, sample_rate=16000, frame_shift=320):
    """Convert time-aligned phonemes to frame labels."""
    frame_duration = frame_shift / sample_rate
    labels = [-100] * num_frames

    for xmin, xmax, phone in intervals:
        normalized = normalize_phoneme(phone)
        if normalized in PHONEME_TO_ID:
            phone_id = PHONEME_TO_ID[normalized]
            start_frame = int(xmin / frame_duration)
            end_frame = int(xmax / frame_duration)
            for f in range(max(0, start_frame), min(num_frames, end_frame)):
                labels[f] = phone_id

    return labels


def get_textgrid_path(audio_path: str) -> str:
    """Get TextGrid path from audio path."""
    parts = audio_path.split('/')
    split = parts[-4]
    speaker_id = parts[-3]
    chapter_id = parts[-2]
    filename = parts[-1].replace('.flac', '.TextGrid').replace('.wav', '.TextGrid')
    return f"{ALIGNMENT_BASE}/{split}/{speaker_id}/{chapter_id}/{filename}"


class LibriSpeechPhonemeDataset(Dataset):
    """LibriSpeech dataset with phoneme labels from TextGrid alignments."""

    def __init__(self, manifest_path: str, max_audio_length: float = 10.0,
                 min_audio_length: float = 2.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.min_samples = int(min_audio_length * sample_rate)

        with open(manifest_path, 'r') as f:
            data = json.load(f)

        self.samples = []
        for item in data:
            item['audio_path'] = reanchor_audio_path(item['audio_path'])
            audio_path = item['audio_path']
            textgrid_path = get_textgrid_path(audio_path)
            if os.path.exists(textgrid_path):
                self.samples.append(item)

        logger.info(f"Loaded {len(self.samples)} samples with alignments from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        audio_path = item['audio_path']
        textgrid_path = get_textgrid_path(audio_path)

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if waveform.shape[0] > self.max_samples:
            start = random.randint(0, waveform.shape[0] - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
            start_time = start / self.sample_rate
        else:
            start_time = 0.0

        intervals = parse_textgrid(textgrid_path)
        num_frames = waveform.shape[0] // 320

        if start_time > 0:
            adjusted_intervals = []
            for xmin, xmax, phone in intervals:
                new_xmin = xmin - start_time
                new_xmax = xmax - start_time
                if new_xmax > 0 and new_xmin < waveform.shape[0] / self.sample_rate:
                    adjusted_intervals.append((max(0, new_xmin), new_xmax, phone))
            intervals = adjusted_intervals

        labels = get_frame_labels(intervals, num_frames)

        return {
            'waveform': waveform,
            'labels': torch.tensor(labels, dtype=torch.long),
            'length': waveform.shape[0],
        }


def collate_fn(batch):
    """Collate with padding for both waveforms and labels."""
    waveforms = [item['waveform'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]

    max_wav_len = max(w.shape[0] for w in waveforms)
    max_lab_len = max(l.shape[0] for l in labels)

    padded_waveforms = []
    padded_labels = []

    for waveform, label in zip(waveforms, labels):
        if waveform.shape[0] < max_wav_len:
            pad = torch.zeros(max_wav_len - waveform.shape[0])
            waveform = torch.cat([waveform, pad])
        padded_waveforms.append(waveform)

        if label.shape[0] < max_lab_len:
            pad = torch.full((max_lab_len - label.shape[0],), -100, dtype=torch.long)
            label = torch.cat([label, pad])
        padded_labels.append(label)

    return {
        'waveform': torch.stack(padded_waveforms),
        'labels': torch.stack(padded_labels),
        'lengths': torch.tensor(lengths),
    }


# === MODEL (same architecture as RVQ supervised, but with BPC head) ===

class VQLayer(nn.Module):
    """Single VQ layer for RVQ stack."""

    def __init__(self, num_codes: int, code_dim: int):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z: torch.Tensor):
        B, T, D = z.shape
        z_flat = z.reshape(-1, D)

        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.codebook.weight.t())
        )

        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices).view(B, T, D)
        indices = indices.view(B, T)

        residual = z - z_q.detach()

        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z)

        z_q = z + (z_q - z).detach()

        encodings = F.one_hot(indices.view(-1), self.num_codes).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        unique_codes = indices.unique().numel()

        return {
            'quantized': z_q,
            'residual': residual,
            'indices': indices,
            'loss_codebook': loss_codebook,
            'loss_commitment': loss_commitment,
            'perplexity': perplexity,
            'unique_codes': unique_codes,
        }


class ResidualVQ(nn.Module):
    """Residual Vector Quantization."""

    def __init__(self, num_layers: int = 4, num_codes: int = 128,
                 code_dim: int = 512, commitment_weight: float = 0.25):
        super().__init__()
        self.num_layers = num_layers
        self.commitment_weight = commitment_weight

        self.vq_layers = nn.ModuleList([
            VQLayer(num_codes, code_dim) for _ in range(num_layers)
        ])

        logger.info(f"ResidualVQ: {num_layers} layers x {num_codes} codes")
        logger.info(f"  Capacity: {num_codes}^{num_layers} = {num_codes**num_layers:,} combinations")

    def forward(self, z: torch.Tensor):
        quantized_sum = torch.zeros_like(z)
        total_loss = 0.0
        residual = z

        perplexities = []
        unique_codes = []
        all_indices = []

        for layer in self.vq_layers:
            out = layer(residual)

            quantized_sum = quantized_sum + out['quantized']
            residual = out['residual']

            layer_loss = out['loss_codebook'] + self.commitment_weight * out['loss_commitment']
            total_loss = total_loss + layer_loss

            perplexities.append(out['perplexity'])
            unique_codes.append(out['unique_codes'])
            all_indices.append(out['indices'])

        return {
            'quantized': quantized_sum,
            'vq_loss': total_loss,
            'perplexities': perplexities,
            'unique_codes': unique_codes,
            'all_indices': all_indices,
            'residual_norm': residual.norm().item(),
        }


class ProtoSpeechBPC(nn.Module):
    """ProtoSpeech with RVQ, trained on Broad Phonetic Classes.

    Two classification heads:
    - bpc_head: trained with CE loss on BPC labels (main objective)
    - fine_head: trained with CE loss on 43-class phoneme labels (auxiliary, weight --fine_weight)
    """

    def __init__(self, hidden_dim: int = 512, num_transformer_layers: int = 6,
                 num_heads: int = 8, num_vq_layers: int = 4, num_codes: int = 128,
                 num_bpc_classes: int = 8, num_fine_classes: int = 43,
                 dropout: float = 0.1):
        super().__init__()

        self.num_bpc_classes = num_bpc_classes
        self.num_fine_classes = num_fine_classes

        # CNN Encoder (as in train_rvq_supervised.py)
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(hidden_dim, hidden_dim), nn.GELU(),
        )

        # Transformer
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            )
            for _ in range(num_transformer_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # RVQ
        self.rvq = ResidualVQ(num_vq_layers, num_codes, hidden_dim)

        # BPC classification head (broad classes)
        self.bpc_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bpc_classes)
        )

        # Fine-grained phoneme head (43-class, auxiliary loss)
        self.fine_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_fine_classes)
        )

        params = sum(p.numel() for p in self.parameters())
        logger.info(f"ProtoSpeechBPC: {params/1e6:.2f}M parameters")
        logger.info(f"  BPC classes: {num_bpc_classes}, Fine classes: {num_fine_classes}")

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)

        rvq_out = self.rvq(hidden)

        bpc_logits = self.bpc_head(rvq_out['quantized'])
        fine_logits = self.fine_head(rvq_out['quantized'])

        return {
            'bpc_logits': bpc_logits,
            'fine_logits': fine_logits,
            'vq_loss': rvq_out['vq_loss'],
            'perplexities': rvq_out['perplexities'],
            'unique_codes': rvq_out['unique_codes'],
            'all_indices': rvq_out['all_indices'],
            'residual_norm': rvq_out['residual_norm'],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpc_type', type=str, default='8class',
                        choices=['8class', '3class_cvs', '3class_vus'])
    parser.add_argument('--num_vq_layers', type=int, default=4)
    parser.add_argument('--num_codes', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vq_weight', type=float, default=1.0)
    parser.add_argument('--fine_weight', type=float, default=0.5,
                        help='Weight for fine-grained phoneme loss (auxiliary)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint output dir. Default: CHECKPOINTS_DIR/rvq_bpc_[<n>layer_]<bpc_type>.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (Python/NumPy/PyTorch). '
                             'When set, also forces cudnn deterministic mode.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Determinism over throughput: needed for reproducible multi-seed runs.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")
    else:
        logger.info("Seed: unseeded (OS entropy)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Determine BPC config
    if args.bpc_type == '8class':
        num_bpc = 8
        idx_to_name = IDX_TO_BPC_8
    elif args.bpc_type == '3class_cvs':
        num_bpc = 3
        idx_to_name = IDX_TO_BPC_3_CVS
    else:
        num_bpc = 3
        idx_to_name = IDX_TO_BPC_3_VUS

    # Build phoneme_id -> bpc_id lookup table
    phoneme_to_bpc = build_phoneme_to_bpc_table(args.bpc_type)
    phoneme_to_bpc_tensor = torch.tensor(phoneme_to_bpc, dtype=torch.long, device=device)

    logger.info("=" * 60)
    logger.info(f"RVQ + BROAD PHONETIC CLASSES ({args.bpc_type})")
    logger.info("=" * 60)
    logger.info(f"Classes: {num_bpc} -> {list(idx_to_name.values())}")
    logger.info(f"Fine weight: {args.fine_weight} (auxiliary phoneme loss)")
    logger.info(f"VOCAB_SIZE (fine): {VOCAB_SIZE}")
    logger.info("=" * 60)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.num_vq_layers != 4:
        output_dir = CHECKPOINTS_DIR / f'rvq_bpc_{args.num_vq_layers}layer_{args.bpc_type}'
    else:
        output_dir = CHECKPOINTS_DIR / f'rvq_bpc_{args.bpc_type}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    model = ProtoSpeechBPC(
        num_vq_layers=args.num_vq_layers,
        num_codes=args.num_codes,
        num_bpc_classes=num_bpc,
        num_fine_classes=VOCAB_SIZE,
    ).to(device)

    # Data: as in train_rvq_supervised.py
    train_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_TRAIN_CLEAN_100), 15.0, 1.0),
        batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_DEV_CLEAN), 15.0, 1.0),
        batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.max_steps, pct_start=0.08)

    global_step = 0
    best_bpc_acc = 0.0
    best_fine_acc = 0.0

    logger.info("Starting training...")

    for epoch in range(10000):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            wav = batch['waveform'].to(device)
            fine_labels = batch['labels'].to(device)  # phoneme_vocab IDs

            out = model(wav)
            bpc_logits = out['bpc_logits']
            fine_logits = out['fine_logits']

            # Align lengths
            min_len = min(bpc_logits.size(1), fine_labels.size(1))
            bpc_logits = bpc_logits[:, :min_len]
            fine_logits = fine_logits[:, :min_len]
            fine_labels_trimmed = fine_labels[:, :min_len]

            # Convert fine labels to BPC labels using lookup table
            # Clamp to valid range; -100 stays as -100 via the lookup table mapping
            safe_labels = fine_labels_trimmed.clone()
            valid_mask_fine = (safe_labels >= 0) & (safe_labels < len(phoneme_to_bpc))
            # Set invalid labels to 0 temporarily for indexing, then override with -100
            safe_labels[~valid_mask_fine] = 0
            bpc_labels = phoneme_to_bpc_tensor[safe_labels]
            bpc_labels[~valid_mask_fine] = -100

            # BPC loss (main)
            valid_bpc = bpc_labels >= 0
            if valid_bpc.sum() == 0:
                continue

            bpc_loss = F.cross_entropy(
                bpc_logits.reshape(-1, num_bpc),
                bpc_labels.reshape(-1),
                ignore_index=-100
            )

            # Fine-grained phoneme loss (auxiliary)
            fine_loss = F.cross_entropy(
                fine_logits.reshape(-1, VOCAB_SIZE),
                fine_labels_trimmed.reshape(-1),
                ignore_index=-100
            )

            total_loss = bpc_loss + args.fine_weight * fine_loss + args.vq_weight * out['vq_loss']

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Codebook utilization logging every 100 steps
            if global_step % 100 == 0:
                with torch.no_grad():
                    dead_per_layer = []
                    for layer_indices in out['all_indices']:
                        flat = layer_indices.reshape(-1)
                        counts = torch.zeros(args.num_codes, device=device)
                        counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
                        avg_freq = flat.numel() / args.num_codes
                        dead = (counts < 0.01 * avg_freq).sum().item()
                        dead_per_layer.append(dead)
                    dead_str = ", ".join([str(d) for d in dead_per_layer])
                    unique_str_100 = ", ".join([str(u) for u in out['unique_codes']])
                    perp_str_100 = ", ".join([f"{p.item():.1f}" for p in out['perplexities']])
                    logger.info(
                        f"Step {global_step} [VQ UTIL] "
                        f"Unique: [{unique_str_100}]/{args.num_codes} | "
                        f"Dead: [{dead_str}] | "
                        f"Perp: [{perp_str_100}]"
                    )

            if global_step % 200 == 0:
                with torch.no_grad():
                    bpc_preds = bpc_logits.argmax(dim=-1)
                    bpc_acc = (bpc_preds[valid_bpc] == bpc_labels[valid_bpc]).float().mean().item()

                    valid_fine = fine_labels_trimmed >= 0
                    fine_preds = fine_logits.argmax(dim=-1)
                    fine_acc = (fine_preds[valid_fine] == fine_labels_trimmed[valid_fine]).float().mean().item()

                perp_str = ", ".join([f"{p.item():.1f}" for p in out['perplexities']])
                unique_str = ", ".join([str(u) for u in out['unique_codes']])

                logger.info(
                    f"Step {global_step} | "
                    f"BPC CE: {bpc_loss.item():.3f} | Fine CE: {fine_loss.item():.3f} | "
                    f"BPC Acc: {bpc_acc:.1%} | Fine Acc: {fine_acc:.1%} | "
                    f"Perp: [{perp_str}] | "
                    f"Codes: [{unique_str}]"
                )

            if global_step % 2000 == 0 and global_step > 0:
                model.eval()
                bpc_correct, fine_correct, total_bpc, total_fine = 0, 0, 0, 0

                with torch.no_grad():
                    for vb in val_loader:
                        vw = vb['waveform'].to(device)
                        vl = vb['labels'].to(device)

                        vout = model(vw)

                        ml = min(vout['bpc_logits'].size(1), vl.size(1))
                        vl_trimmed = vl[:, :ml]

                        # Convert to BPC labels
                        safe_vl = vl_trimmed.clone()
                        valid_vl = (safe_vl >= 0) & (safe_vl < len(phoneme_to_bpc))
                        safe_vl[~valid_vl] = 0
                        vbpc_labels = phoneme_to_bpc_tensor[safe_vl]
                        vbpc_labels[~valid_vl] = -100

                        # BPC accuracy
                        bpc_mask = vbpc_labels >= 0
                        bpc_preds = vout['bpc_logits'][:, :ml].argmax(dim=-1)
                        bpc_correct += (bpc_preds[bpc_mask] == vbpc_labels[bpc_mask]).sum().item()
                        total_bpc += bpc_mask.sum().item()

                        # Fine accuracy
                        fine_mask = vl_trimmed >= 0
                        fine_preds = vout['fine_logits'][:, :ml].argmax(dim=-1)
                        fine_correct += (fine_preds[fine_mask] == vl_trimmed[fine_mask]).sum().item()
                        total_fine += fine_mask.sum().item()

                val_bpc_acc = bpc_correct / total_bpc if total_bpc > 0 else 0
                val_fine_acc = fine_correct / total_fine if total_fine > 0 else 0

                logger.info(
                    f"[VAL] Step {global_step} | "
                    f"BPC Acc: {val_bpc_acc:.2%} | Fine Acc: {val_fine_acc:.2%}"
                )

                if val_bpc_acc > best_bpc_acc:
                    best_bpc_acc = val_bpc_acc
                    best_fine_acc = val_fine_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'step': global_step,
                        'bpc_acc': val_bpc_acc,
                        'fine_acc': val_fine_acc,
                        'bpc_type': args.bpc_type,
                    }, output_dir / 'best.pt')
                    logger.info(f"Saved best model (BPC={val_bpc_acc:.2%}, Fine={val_fine_acc:.2%})")

                model.train()

            global_step += 1

        if global_step >= args.max_steps:
            break

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'best_bpc_acc': best_bpc_acc,
        'best_fine_acc': best_fine_acc,
        'bpc_type': args.bpc_type,
    }, output_dir / 'final.pt')

    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"BPC Type: {args.bpc_type} ({num_bpc} classes)")
    logger.info(f"Best BPC Acc: {best_bpc_acc:.2%}")
    logger.info(f"Best Fine Acc: {best_fine_acc:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
