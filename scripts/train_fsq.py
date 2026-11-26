#!/usr/bin/env python3
"""
FSQ (Finite Scalar Quantization) training for ProtoSpeech.

FSQ has no learned codebook: its implicit codes are the points of the level
grid, for example [8,5,5,5] = 1000 or [8,8,5,5,5] = 8000 codes, and it is
designed to use nearly all of them.

Data pipeline: identical to train_rvq_bpc.py.
Architecture:  identical CNN encoder + Transformer.
Only change:   the ResidualVQ is replaced by the FSQ quantizer.

Supports:
  --fsq_levels "8,5,5,5"   -> 1000 implicit codes
  --fsq_levels "8,8,5,5,5" -> 8000 implicit codes
  --bpc_supervision         -> dual-head BPC + fine (as in train_rvq_bpc.py)
"""
import os
import re
import json
import math
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
    IDX_TO_BPC_8, BPC_8_TO_IDX,
)
from models.fsq import FSQ

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


# === MODEL ===

class ProtoSpeechFSQ(nn.Module):
    """ProtoSpeech with FSQ quantizer.

    Architecture: CNN encoder -> Transformer -> FSQ -> classification head(s)

    When bpc_supervision=True, uses dual heads (BPC + fine) like ProtoSpeechBPC.
    When bpc_supervision=False, uses single fine-grained phoneme head.
    """

    def __init__(self, hidden_dim: int = 512, num_transformer_layers: int = 6,
                 num_heads: int = 8, fsq_levels: list = None,
                 num_bpc_classes: int = 8, num_fine_classes: int = 43,
                 bpc_supervision: bool = False, dropout: float = 0.1):
        super().__init__()

        if fsq_levels is None:
            fsq_levels = [8, 5, 5, 5]

        self.bpc_supervision = bpc_supervision
        self.num_bpc_classes = num_bpc_classes
        self.num_fine_classes = num_fine_classes

        # CNN Encoder (as in train_rvq_bpc.py)
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

        # Transformer (as in train_rvq_bpc.py)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            )
            for _ in range(num_transformer_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # FSQ quantizer (replaces ResidualVQ)
        self.fsq = FSQ(input_dim=hidden_dim, levels=fsq_levels)

        # Classification heads
        self.fine_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_fine_classes)
        )

        if bpc_supervision:
            self.bpc_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_bpc_classes)
            )

        params = sum(p.numel() for p in self.parameters())
        logger.info(f"ProtoSpeechFSQ: {params/1e6:.2f}M parameters")
        logger.info(f"  FSQ levels: {fsq_levels} -> {math.prod(fsq_levels)} implicit codes")
        logger.info(f"  BPC supervision: {bpc_supervision}")

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)

        fsq_out = self.fsq(hidden)

        result = {
            'fine_logits': self.fine_head(fsq_out['quantized']),
            'vq_loss': fsq_out['vq_loss'],
            'perplexities': fsq_out['perplexities'],
            'unique_codes': fsq_out['unique_codes'],
            'all_indices': fsq_out['all_indices'],
            'codes': fsq_out['codes'],
            'residual_norm': fsq_out['residual_norm'],
        }

        if self.bpc_supervision:
            result['bpc_logits'] = self.bpc_head(fsq_out['quantized'])

        return result


def main():
    parser = argparse.ArgumentParser(description='Train ProtoSpeech with FSQ quantizer')
    parser.add_argument('--fsq_levels', type=str, default='8,5,5,5',
                        help='Comma-separated FSQ levels (e.g. "8,5,5,5" for 1K codes)')
    parser.add_argument('--bpc_supervision', action='store_true',
                        help='Enable dual-head BPC + fine supervision')
    parser.add_argument('--bpc_type', type=str, default='8class',
                        choices=['8class'])
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fine_weight', type=float, default=0.5,
                        help='Weight for fine-grained phoneme loss when BPC is active')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint output dir. Default: CHECKPOINTS_DIR/fsq_<levels>_<num_codes>[_bpc].')
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

    # Parse FSQ levels
    fsq_levels = [int(x) for x in args.fsq_levels.split(',')]
    num_codes = math.prod(fsq_levels)
    levels_str = 'x'.join(str(l) for l in fsq_levels)

    # BPC config
    num_bpc = 8
    idx_to_name = IDX_TO_BPC_8
    phoneme_to_bpc = build_phoneme_to_bpc_table(args.bpc_type)
    phoneme_to_bpc_tensor = torch.tensor(phoneme_to_bpc, dtype=torch.long, device=device)

    logger.info("=" * 60)
    logger.info(f"FSQ TRAINING")
    logger.info("=" * 60)
    logger.info(f"Levels: {fsq_levels} -> {num_codes} implicit codes")
    logger.info(f"BPC supervision: {args.bpc_supervision}")
    if args.bpc_supervision:
        logger.info(f"BPC classes: {num_bpc} -> {list(idx_to_name.values())}")
        logger.info(f"Fine weight: {args.fine_weight}")
    logger.info(f"VOCAB_SIZE (fine): {VOCAB_SIZE}")
    logger.info("=" * 60)

    # Output directory: --output_dir overrides the auto-derived default.
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"fsq_{levels_str}_{num_codes}"
        if args.bpc_supervision:
            suffix += "_bpc"
        output_dir = CHECKPOINTS_DIR / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    model = ProtoSpeechFSQ(
        fsq_levels=fsq_levels,
        num_bpc_classes=num_bpc,
        num_fine_classes=VOCAB_SIZE,
        bpc_supervision=args.bpc_supervision,
    ).to(device)

    # Data: as in train_rvq_bpc.py
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

    # Track epoch-level code usage (skip full histogram for very large codebooks)
    track_code_histogram = (num_codes <= 100000)
    if track_code_histogram:
        epoch_code_counts = torch.zeros(num_codes, device=device)

    logger.info("Starting training...")

    for epoch in range(10000):
        model.train()
        if track_code_histogram:
            epoch_code_counts.zero_()

        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            wav = batch['waveform'].to(device)
            fine_labels = batch['labels'].to(device)

            out = model(wav)
            fine_logits = out['fine_logits']

            # Align lengths
            min_len = min(fine_logits.size(1), fine_labels.size(1))
            fine_logits = fine_logits[:, :min_len]
            fine_labels_trimmed = fine_labels[:, :min_len]

            # Fine-grained phoneme loss
            fine_loss = F.cross_entropy(
                fine_logits.reshape(-1, VOCAB_SIZE),
                fine_labels_trimmed.reshape(-1),
                ignore_index=-100
            )

            if args.bpc_supervision:
                bpc_logits = out['bpc_logits'][:, :min_len]

                # Convert fine labels to BPC labels
                safe_labels = fine_labels_trimmed.clone()
                valid_mask_fine = (safe_labels >= 0) & (safe_labels < len(phoneme_to_bpc))
                safe_labels[~valid_mask_fine] = 0
                bpc_labels = phoneme_to_bpc_tensor[safe_labels]
                bpc_labels[~valid_mask_fine] = -100

                valid_bpc = bpc_labels >= 0
                if valid_bpc.sum() == 0:
                    continue

                bpc_loss = F.cross_entropy(
                    bpc_logits.reshape(-1, num_bpc),
                    bpc_labels.reshape(-1),
                    ignore_index=-100
                )

                total_loss = bpc_loss + args.fine_weight * fine_loss
            else:
                total_loss = fine_loss

            # No vq_loss term: FSQ has no codebook loss

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Accumulate code usage for epoch histogram
            if track_code_histogram:
                with torch.no_grad():
                    batch_indices = out['all_indices'][0].reshape(-1)
                    epoch_code_counts.scatter_add_(
                        0, batch_indices,
                        torch.ones_like(batch_indices, dtype=torch.float32)
                    )

            # Logging every 100 steps (codebook utilization)
            if global_step % 100 == 0:
                with torch.no_grad():
                    # Code utilization stats (use codes when indices overflow int64)
                    if out['all_indices'][0] is not None:
                        util = model.fsq.get_codebook_utilization(indices=out['all_indices'][0])
                    else:
                        util = model.fsq.get_codebook_utilization(codes=out['codes'])

                    logger.info(
                        f"Step {global_step} | "
                        f"Codes: {util['unique_codes']}/{num_codes} "
                        f"({util['utilization_pct']:.1f}%) | "
                        f"Dead: {util['num_dead']} | "
                        f"Perp: {out['perplexities'][0].item():.1f} | "
                        f"Entropy: {util['entropy']:.2f}/{util['max_entropy']:.2f} "
                        f"({util['entropy_ratio']:.1%})"
                    )

            # Accuracy logging every 200 steps
            if global_step % 200 == 0:
                with torch.no_grad():
                    valid_fine = fine_labels_trimmed >= 0
                    fine_preds = fine_logits.argmax(dim=-1)
                    fine_acc = (fine_preds[valid_fine] == fine_labels_trimmed[valid_fine]).float().mean().item()

                    perp_str = f"{out['perplexities'][0].item():.1f}"
                    unique_str = str(out['unique_codes'][0])

                    log_msg = (
                        f"Step {global_step} | "
                        f"Fine CE: {fine_loss.item():.3f} | "
                        f"Fine Acc: {fine_acc:.1%} | "
                        f"Perp: {perp_str} | "
                        f"Codes: {unique_str}"
                    )

                    if args.bpc_supervision:
                        bpc_preds = bpc_logits.argmax(dim=-1)
                        bpc_acc = (bpc_preds[valid_bpc] == bpc_labels[valid_bpc]).float().mean().item()
                        log_msg = (
                            f"Step {global_step} | "
                            f"BPC CE: {bpc_loss.item():.3f} | Fine CE: {fine_loss.item():.3f} | "
                            f"BPC Acc: {bpc_acc:.1%} | Fine Acc: {fine_acc:.1%} | "
                            f"Perp: {perp_str} | "
                            f"Codes: {unique_str}"
                        )

                    logger.info(log_msg)

            # Validation every 2000 steps
            if global_step % 2000 == 0 and global_step > 0:
                model.eval()
                fine_correct, total_fine = 0, 0
                bpc_correct, total_bpc = 0, 0

                with torch.no_grad():
                    for vb in val_loader:
                        vw = vb['waveform'].to(device)
                        vl = vb['labels'].to(device)

                        vout = model(vw)

                        ml = min(vout['fine_logits'].size(1), vl.size(1))
                        vl_trimmed = vl[:, :ml]

                        # Fine accuracy
                        fine_mask = vl_trimmed >= 0
                        fine_preds = vout['fine_logits'][:, :ml].argmax(dim=-1)
                        fine_correct += (fine_preds[fine_mask] == vl_trimmed[fine_mask]).sum().item()
                        total_fine += fine_mask.sum().item()

                        if args.bpc_supervision:
                            # BPC accuracy
                            safe_vl = vl_trimmed.clone()
                            valid_vl = (safe_vl >= 0) & (safe_vl < len(phoneme_to_bpc))
                            safe_vl[~valid_vl] = 0
                            vbpc_labels = phoneme_to_bpc_tensor[safe_vl]
                            vbpc_labels[~valid_vl] = -100

                            bpc_mask = vbpc_labels >= 0
                            bpc_preds = vout['bpc_logits'][:, :ml].argmax(dim=-1)
                            bpc_correct += (bpc_preds[bpc_mask] == vbpc_labels[bpc_mask]).sum().item()
                            total_bpc += bpc_mask.sum().item()

                val_fine_acc = fine_correct / total_fine if total_fine > 0 else 0

                val_msg = f"[VAL] Step {global_step} | Fine Acc: {val_fine_acc:.2%}"
                if args.bpc_supervision:
                    val_bpc_acc = bpc_correct / total_bpc if total_bpc > 0 else 0
                    val_msg += f" | BPC Acc: {val_bpc_acc:.2%}"

                logger.info(val_msg)

                if val_fine_acc > best_fine_acc:
                    best_fine_acc = val_fine_acc
                    save_dict = {
                        'model_state_dict': model.state_dict(),
                        'step': global_step,
                        'fine_acc': val_fine_acc,
                        'fsq_levels': fsq_levels,
                        'num_codes': num_codes,
                        'bpc_supervision': args.bpc_supervision,
                    }
                    if args.bpc_supervision:
                        save_dict['bpc_acc'] = val_bpc_acc
                        best_bpc_acc = val_bpc_acc
                    torch.save(save_dict, output_dir / 'best.pt')
                    logger.info(f"Saved best model (Fine={val_fine_acc:.2%})")

                model.train()

            global_step += 1

        if global_step >= args.max_steps:
            break

        # Epoch-level code usage stats
        if track_code_histogram:
            used = (epoch_code_counts > 0).sum().item()
            avg_freq = epoch_code_counts.sum().item() / num_codes
            dead = (epoch_code_counts < 0.01 * avg_freq).sum().item()
            logger.info(
                f"[EPOCH {epoch}] Code usage: {used}/{num_codes} "
                f"({used/num_codes*100:.1f}%), Dead: {dead}"
            )
        else:
            logger.info(f"[EPOCH {epoch}] Large codebook ({num_codes}): see per-step logs")

    # Final save
    save_dict = {
        'model_state_dict': model.state_dict(),
        'step': global_step,
        'best_fine_acc': best_fine_acc,
        'fsq_levels': fsq_levels,
        'num_codes': num_codes,
        'bpc_supervision': args.bpc_supervision,
    }
    if args.bpc_supervision:
        save_dict['best_bpc_acc'] = best_bpc_acc
    torch.save(save_dict, output_dir / 'final.pt')

    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"FSQ levels: {fsq_levels} -> {num_codes} implicit codes")
    logger.info(f"Best Fine Acc: {best_fine_acc:.2%}")
    if args.bpc_supervision:
        logger.info(f"Best BPC Acc: {best_bpc_acc:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
