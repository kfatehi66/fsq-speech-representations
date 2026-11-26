#!/usr/bin/env python3
"""
Hierarchical 2-Layer RVQ: Layer 1 for BPC, Layer 2 for fine-grained refinement.

Architecture:
  Waveform -> CNN -> Transformer -> VQ Layer1 -> BPC head (coarse)
  residual of VQ Layer1 -> VQ Layer2
  [L1 + L2 sum] -> Fine head (43 phonemes)

Loss = α * BPC_loss(L1) + β * Fine_loss(L1+L2) + VQ_loss

The BPC loss on Layer 1 alone pushes Layer 1 towards coarse phonetic
structure; the fine loss on L1+L2 combined lets Layer 2 learn within-class
refinements.
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
from models.checkpoint_fingerprint import build_fingerprint
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
    parts = audio_path.split('/')
    split = parts[-4]
    speaker_id = parts[-3]
    chapter_id = parts[-2]
    filename = parts[-1].replace('.flac', '.TextGrid').replace('.wav', '.TextGrid')
    return f"{ALIGNMENT_BASE}/{split}/{speaker_id}/{chapter_id}/{filename}"


class LibriSpeechPhonemeDataset(Dataset):
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


# === MODEL ===

class VQLayer(nn.Module):
    """Single VQ layer."""

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

        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z)

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        encodings = F.one_hot(indices.view(-1), self.num_codes).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        unique_codes = indices.unique().numel()

        return {
            'quantized': z_q_st,
            'indices': indices,
            'loss_codebook': loss_codebook,
            'loss_commitment': loss_commitment,
            'perplexity': perplexity,
            'unique_codes': unique_codes,
        }


class ProtoSpeechHierarchical(nn.Module):
    """Hierarchical 2-Layer RVQ.

    Layer 1: encodes the broad phonetic class (BPC), the coarse structure.
    Layer 2: encodes the within-class refinement, the fine-grained phonemes.

    BPC head reads Layer 1 output only.
    Fine head reads Layer 1 + Layer 2 combined (sum).
    """

    def __init__(self, hidden_dim: int = 512, num_transformer_layers: int = 6,
                 num_heads: int = 8, num_codes: int = 128,
                 num_bpc_classes: int = 8, num_fine_classes: int = 43,
                 commitment_weight: float = 0.25, dropout: float = 0.1):
        super().__init__()

        self.commitment_weight = commitment_weight

        # CNN Encoder (as in the other trainers)
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

        # Two separate VQ layers (not ResidualVQ, because the heads need per-layer access)
        self.vq_layer1 = VQLayer(num_codes, hidden_dim)
        self.vq_layer2 = VQLayer(num_codes, hidden_dim)

        # BPC head: reads the Layer 1 output only
        self.bpc_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bpc_classes)
        )

        # Fine head: reads Layer 1 + Layer 2 (sum)
        self.fine_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_fine_classes)
        )

        params = sum(p.numel() for p in self.parameters())
        logger.info(f"ProtoSpeechHierarchical: {params/1e6:.2f}M parameters")
        logger.info(f"  Layer 1 -> BPC head ({num_bpc_classes} classes)")
        logger.info(f"  Layer 1+2 -> Fine head ({num_fine_classes} classes)")
        logger.info(f"  Codebook: 2 layers x {num_codes} codes = {num_codes**2:,} combinations")

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)

        # Layer 1: quantize continuous features
        vq1_out = self.vq_layer1(hidden)
        z1 = vq1_out['quantized']

        # Layer 2: quantize residual (what Layer 1 couldn't capture)
        residual = hidden - z1
        vq2_out = self.vq_layer2(residual)
        z2 = vq2_out['quantized']

        # BPC prediction from Layer 1 only
        bpc_logits = self.bpc_head(z1)

        # Fine prediction from Layer 1 + Layer 2 (standard RVQ sum)
        combined = z1 + z2
        fine_logits = self.fine_head(combined)

        # VQ losses
        vq_loss = (
            vq1_out['loss_codebook'] + self.commitment_weight * vq1_out['loss_commitment']
            + vq2_out['loss_codebook'] + self.commitment_weight * vq2_out['loss_commitment']
        )

        return {
            'bpc_logits': bpc_logits,
            'fine_logits': fine_logits,
            'vq_loss': vq_loss,
            'perplexities': [vq1_out['perplexity'], vq2_out['perplexity']],
            'unique_codes': [vq1_out['unique_codes'], vq2_out['unique_codes']],
            'all_indices': [vq1_out['indices'], vq2_out['indices']],
            'residual_norm': residual.norm().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bpc_type', type=str, default='8class',
                        choices=['8class', '3class_cvs', '3class_vus'])
    parser.add_argument('--num_codes', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--vq_weight', type=float, default=1.0)
    parser.add_argument('--bpc_weight', type=float, default=1.0,
                        help='Weight for BPC loss on Layer 1 (alpha)')
    parser.add_argument('--fine_weight', type=float, default=1.0,
                        help='Weight for fine-grained loss on Layer 1+2 (beta)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint output dir. Default: CHECKPOINTS_DIR/rvq_hierarchical_<bpc_type>.')
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

    # BPC config
    if args.bpc_type == '8class':
        num_bpc = 8
        idx_to_name = IDX_TO_BPC_8
    elif args.bpc_type == '3class_cvs':
        num_bpc = 3
        idx_to_name = IDX_TO_BPC_3_CVS
    else:
        num_bpc = 3
        idx_to_name = IDX_TO_BPC_3_VUS

    phoneme_to_bpc = build_phoneme_to_bpc_table(args.bpc_type)
    phoneme_to_bpc_tensor = torch.tensor(phoneme_to_bpc, dtype=torch.long, device=device)

    logger.info("=" * 60)
    logger.info(f"HIERARCHICAL 2-LAYER RVQ ({args.bpc_type})")
    logger.info("=" * 60)
    logger.info(f"Layer 1 -> BPC: {num_bpc} classes ({list(idx_to_name.values())})")
    logger.info(f"Layer 1+2 -> Fine: {VOCAB_SIZE} classes")
    logger.info(f"BPC weight (α): {args.bpc_weight}")
    logger.info(f"Fine weight (β): {args.fine_weight}")
    logger.info("=" * 60)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.fine_weight != 1.0 or args.bpc_weight != 1.0:
        output_dir = CHECKPOINTS_DIR / f'rvq_hierarchical_a{args.bpc_weight:.0f}_b{args.fine_weight:.0f}_{args.bpc_type}'
    else:
        output_dir = CHECKPOINTS_DIR / f'rvq_hierarchical_{args.bpc_type}'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    model = ProtoSpeechHierarchical(
        num_codes=args.num_codes,
        num_bpc_classes=num_bpc,
        num_fine_classes=VOCAB_SIZE,
    ).to(device)

    # Record the configuration that was built. preserve_symmetry,
    # extract_layers and similar options change what the weights mean without
    # changing their shape, so load_state_dict cannot detect a mismatch; the
    # loader rebuilds the model from this record and checks it against it.
    FINGERPRINT = build_fingerprint(model, args, num_codes=args.num_codes,
                                    num_bpc_classes=num_bpc, hidden_dim=512)

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
    best_fine_acc = 0.0
    best_bpc_acc = 0.0

    logger.info("Starting training...")

    for epoch in range(10000):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            wav = batch['waveform'].to(device)
            fine_labels = batch['labels'].to(device)

            out = model(wav)
            bpc_logits = out['bpc_logits']
            fine_logits = out['fine_logits']

            # Align lengths
            min_len = min(bpc_logits.size(1), fine_labels.size(1))
            bpc_logits = bpc_logits[:, :min_len]
            fine_logits = fine_logits[:, :min_len]
            fine_labels_trimmed = fine_labels[:, :min_len]

            # Convert fine labels to BPC labels
            safe_labels = fine_labels_trimmed.clone()
            valid_mask_fine = (safe_labels >= 0) & (safe_labels < len(phoneme_to_bpc))
            safe_labels[~valid_mask_fine] = 0
            bpc_labels = phoneme_to_bpc_tensor[safe_labels]
            bpc_labels[~valid_mask_fine] = -100

            valid_bpc = bpc_labels >= 0
            if valid_bpc.sum() == 0:
                continue

            # BPC loss on Layer 1 only
            bpc_loss = F.cross_entropy(
                bpc_logits.reshape(-1, num_bpc),
                bpc_labels.reshape(-1),
                ignore_index=-100
            )

            # Fine loss on Layer 1+2 combined
            fine_loss = F.cross_entropy(
                fine_logits.reshape(-1, VOCAB_SIZE),
                fine_labels_trimmed.reshape(-1),
                ignore_index=-100
            )

            total_loss = (
                args.bpc_weight * bpc_loss
                + args.fine_weight * fine_loss
                + args.vq_weight * out['vq_loss']
            )

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

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
                    f"BPC(L1): {bpc_loss.item():.3f} | Fine(L1+L2): {fine_loss.item():.3f} | "
                    f"BPC Acc: {bpc_acc:.1%} | Fine Acc: {fine_acc:.1%} | "
                    f"Perp: [{perp_str}] | Codes: [{unique_str}]"
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

                        safe_vl = vl_trimmed.clone()
                        valid_vl = (safe_vl >= 0) & (safe_vl < len(phoneme_to_bpc))
                        safe_vl[~valid_vl] = 0
                        vbpc_labels = phoneme_to_bpc_tensor[safe_vl]
                        vbpc_labels[~valid_vl] = -100

                        bpc_mask = vbpc_labels >= 0
                        bpc_preds = vout['bpc_logits'][:, :ml].argmax(dim=-1)
                        bpc_correct += (bpc_preds[bpc_mask] == vbpc_labels[bpc_mask]).sum().item()
                        total_bpc += bpc_mask.sum().item()

                        fine_mask = vl_trimmed >= 0
                        fine_preds = vout['fine_logits'][:, :ml].argmax(dim=-1)
                        fine_correct += (fine_preds[fine_mask] == vl_trimmed[fine_mask]).sum().item()
                        total_fine += fine_mask.sum().item()

                val_bpc_acc = bpc_correct / total_bpc if total_bpc > 0 else 0
                val_fine_acc = fine_correct / total_fine if total_fine > 0 else 0

                logger.info(
                    f"[VAL] Step {global_step} | "
                    f"BPC Acc(L1): {val_bpc_acc:.2%} | Fine Acc(L1+L2): {val_fine_acc:.2%}"
                )

                if val_fine_acc > best_fine_acc:
                    best_fine_acc = val_fine_acc
                    best_bpc_acc = val_bpc_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config_fingerprint': FINGERPRINT,
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
        'config_fingerprint': FINGERPRINT,
        'step': global_step,
        'best_bpc_acc': best_bpc_acc,
        'best_fine_acc': best_fine_acc,
        'bpc_type': args.bpc_type,
    }, output_dir / 'final.pt')

    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"BPC Type: {args.bpc_type} ({num_bpc} classes)")
    logger.info(f"Best BPC Acc (Layer 1): {best_bpc_acc:.2%}")
    logger.info(f"Best Fine Acc (Layer 1+2): {best_fine_acc:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
