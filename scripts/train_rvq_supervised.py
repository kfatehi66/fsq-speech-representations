#!/usr/bin/env python3
"""
ProtoSpeech with Residual Vector Quantization (RVQ), trained on phoneme labels.

With --num_vq_layers L and --num_codes K the RVQ has K^L code combinations,
for example 128^4 = 268M for 4 layers of 128 codes.
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

# The same imports as the fusion baseline trainers
from data.phoneme_vocab import PHONEME_TO_ID, normalize_phoneme, VOCAB_SIZE
from data.dataset import reanchor_audio_path
from models.checkpoint_fingerprint import build_fingerprint

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

        # Filter to only samples with TextGrid alignments
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

        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Crop/pad to max length
        if waveform.shape[0] > self.max_samples:
            start = random.randint(0, waveform.shape[0] - self.max_samples)
            waveform = waveform[start:start + self.max_samples]
            start_time = start / self.sample_rate
        else:
            start_time = 0.0

        # Parse TextGrid
        intervals = parse_textgrid(textgrid_path)

        # Estimate number of frames (CNN downsamples by ~320)
        num_frames = waveform.shape[0] // 320

        # Adjust intervals for cropping
        if start_time > 0:
            adjusted_intervals = []
            for xmin, xmax, phone in intervals:
                new_xmin = xmin - start_time
                new_xmax = xmax - start_time
                if new_xmax > 0 and new_xmin < waveform.shape[0] / self.sample_rate:
                    adjusted_intervals.append((max(0, new_xmin), new_xmax, phone))
            intervals = adjusted_intervals

        # Get frame labels
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
        # Pad waveform
        if waveform.shape[0] < max_wav_len:
            pad = torch.zeros(max_wav_len - waveform.shape[0])
            waveform = torch.cat([waveform, pad])
        padded_waveforms.append(waveform)

        # Pad labels with -100 (ignore index)
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

        # L2 distances to codebook
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.codebook.weight.t())
        )

        # Hard assignment
        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices).view(B, T, D)
        indices = indices.view(B, T)

        # Residual for next layer
        residual = z - z_q.detach()

        # Losses (before STE)
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z)

        # STE: gradient flows through z_q
        z_q = z + (z_q - z).detach()

        # Perplexity
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
    """
    Residual Vector Quantization.
    Each layer quantizes the residual from the previous layer.
    Output = sum of all quantized vectors.
    """

    def __init__(
        self,
        num_layers: int = 4,
        num_codes: int = 128,
        code_dim: int = 512,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.commitment_weight = commitment_weight

        self.vq_layers = nn.ModuleList([
            VQLayer(num_codes, code_dim) for _ in range(num_layers)
        ])

        logger.info(f"ResidualVQ: {num_layers} layers × {num_codes} codes")
        logger.info(f"  Capacity: {num_codes}^{num_layers} = {num_codes**num_layers:,} combinations")

    def forward(self, z: torch.Tensor):
        quantized_sum = torch.zeros_like(z)
        total_loss = 0.0
        residual = z

        perplexities = []
        unique_codes = []

        for layer in self.vq_layers:
            out = layer(residual)

            quantized_sum = quantized_sum + out['quantized']
            residual = out['residual']

            layer_loss = out['loss_codebook'] + self.commitment_weight * out['loss_commitment']
            total_loss = total_loss + layer_loss

            perplexities.append(out['perplexity'])
            unique_codes.append(out['unique_codes'])

        return {
            'quantized': quantized_sum,
            'vq_loss': total_loss,
            'perplexities': perplexities,
            'unique_codes': unique_codes,
            'residual_norm': residual.norm().item(),
        }


class ProtoSpeechRVQ(nn.Module):
    """ProtoSpeech with RVQ bottleneck."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_transformer_layers: int = 6,
        num_heads: int = 8,
        num_vq_layers: int = 4,
        num_codes: int = 128,
        num_phonemes: int = 43,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_phonemes = num_phonemes

        # CNN Encoder
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

        # Phoneme head
        self.phoneme_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phonemes)
        )

        params = sum(p.numel() for p in self.parameters())
        logger.info(f"ProtoSpeechRVQ: {params/1e6:.2f}M parameters")

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)

        # RVQ
        rvq_out = self.rvq(hidden)

        # Classify
        logits = self.phoneme_head(rvq_out['quantized'])

        return {
            'logits': logits,
            'vq_loss': rvq_out['vq_loss'],
            'perplexities': rvq_out['perplexities'],
            'unique_codes': rvq_out['unique_codes'],
            'residual_norm': rvq_out['residual_norm'],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_vq_layers', type=int, default=4)
    parser.add_argument('--num_codes', type=int, default=128)
    parser.add_argument('--vq_weight', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint output dir. Default: CHECKPOINTS_DIR/rvq_supervised.')
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

    logger.info("=" * 60)
    logger.info("PROTOSPEECH WITH RESIDUAL VQ")
    logger.info("=" * 60)
    logger.info(f"RVQ: {args.num_vq_layers} layers × {args.num_codes} codes")
    logger.info(f"Capacity: {args.num_codes}^{args.num_vq_layers} = {args.num_codes**args.num_vq_layers:,}")
    logger.info(f"VOCAB_SIZE: {VOCAB_SIZE}")
    logger.info("=" * 60)

    output_dir = (Path(args.output_dir) if args.output_dir
                  else CHECKPOINTS_DIR / 'rvq_supervised')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    model = ProtoSpeechRVQ(
        num_vq_layers=args.num_vq_layers,
        num_codes=args.num_codes,
        num_phonemes=VOCAB_SIZE,
    ).to(device)

    # Record the configuration that was built. preserve_symmetry,
    # extract_layers and similar options change what the weights mean without
    # changing their shape, so load_state_dict cannot detect a mismatch; the
    # loader rebuilds the model from this record and checks it against it.
    FINGERPRINT = build_fingerprint(model, args, num_vq_layers=args.num_vq_layers,
                                    num_codes=args.num_codes, hidden_dim=512)

    # Data: as in the other supervised trainers
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
    best_acc = 0.0

    logger.info("Starting training...")

    for epoch in range(10000):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            wav = batch['waveform'].to(device)
            labels = batch['labels'].to(device)

            out = model(wav)
            logits = out['logits']

            # Align lengths
            min_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_len]
            labels_trimmed = labels[:, :min_len]

            # Mask invalid
            valid_mask = labels_trimmed >= 0
            if valid_mask.sum() == 0:
                continue

            # Loss
            ce_loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels_trimmed.reshape(-1),
                ignore_index=-100
            )

            total_loss = ce_loss + args.vq_weight * out['vq_loss']

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 200 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds[valid_mask] == labels_trimmed[valid_mask]).float().mean().item()

                perp_str = ", ".join([f"{p.item():.1f}" for p in out['perplexities']])
                unique_str = ", ".join([str(u) for u in out['unique_codes']])

                logger.info(
                    f"Step {global_step} | "
                    f"CE: {ce_loss.item():.3f} | "
                    f"VQ: {out['vq_loss'].item():.3f} | "
                    f"Acc: {acc:.1%} | "
                    f"Perp: [{perp_str}] | "
                    f"Codes: [{unique_str}]"
                )

            if global_step % 2000 == 0 and global_step > 0:
                model.eval()
                correct, total = 0, 0

                with torch.no_grad():
                    for vb in val_loader:
                        vw = vb['waveform'].to(device)
                        vl = vb['labels'].to(device)
                        vout = model(vw)
                        vlogits = vout['logits']

                        ml = min(vlogits.size(1), vl.size(1))
                        mask = vl[:, :ml] >= 0
                        vpreds = vlogits[:, :ml].argmax(dim=-1)
                        correct += (vpreds[mask] == vl[:, :ml][mask]).sum().item()
                        total += mask.sum().item()

                acc = correct / total if total > 0 else 0
                logger.info(f"[VAL] Step {global_step} | Acc: {acc:.2%}")

                if acc > best_acc:
                    best_acc = acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config_fingerprint': FINGERPRINT,
                        'step': global_step,
                        'val_acc': acc,
                    }, output_dir / 'best.pt')
                    logger.info(f"Saved best model (acc={acc:.2%})")

                model.train()

            global_step += 1

        if global_step >= args.max_steps:
            break

    # Final save
    torch.save({
        'model_state_dict': model.state_dict(),
        'config_fingerprint': FINGERPRINT,
        'step': global_step,
        'best_val_acc': best_acc,
    }, output_dir / 'final.pt')

    logger.info("=" * 60)
    logger.info(f"Training complete! Best val acc: {best_acc:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
