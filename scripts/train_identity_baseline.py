#!/usr/bin/env python3
"""
Identity baseline: CNN -> Transformer -> classifier, with no quantiser.

This baseline measures the cost of the bottleneck. The two fusion baselines
without VQ do not measure it:

  * Concat (train_fusion_phoneme_supervised.py) concatenates three Transformer
    layers into a 1536-d representation: layers 4, 5 and 6 in the runs of the
    paper, set with --extract_layers (the trainer's default is 2, 4 and 6).
    That is three times the width of the quantised models, and also a
    different depth, so its advantage over FSQ/RVQ confounds "no bottleneck"
    with "3x wider, multi-layer input".
  * Gated (train_gated_supervised.py) learns a gated fusion over layers 4, 5
    and 6, which that trainer fixes: the same confound plus extra parameters.

This file takes the quantised architecture and removes exactly one thing: the
quantiser. Everything else (CNN, 6-layer Transformer, final LayerNorm, head
width, data pipeline, optimiser, schedule, seed handling) is identical to
scripts/train_fsq.py. So `identity - FSQ` isolates the cost of quantisation,
and `Concat - identity` isolates the benefit of multi-layer fusion.

Differences from scripts/train_fsq.py
-------------------------------------
This file is a copy of scripts/train_fsq.py with these differences:

  1. `from models.fsq import FSQ` removed; no quantiser is constructed.
  2. `ProtoSpeechFSQ` is renamed `ProtoSpeechIdentity`. `self.fsq` is gone;
     forward() feeds the 512-d Transformer output (post-LayerNorm) straight to
     the heads, where train_fsq.py feeds `fsq_out['quantized']`. Nothing else
     in the class changes: same CNN stack, same 6 layers, same head shapes.
  3. forward() no longer returns vq_loss / perplexities / unique_codes /
     all_indices / codes / residual_norm. There are no discrete codes to report.
  4. For the same reason the training loop has no epoch code histogram
     (`track_code_histogram`, `epoch_code_counts`), no every-100-step codebook
     utilisation log, and no Perp/Codes fields in the every-200-step log line.
     The loss is the same as in train_fsq.py, which adds no vq_loss term.
  5. `--fsq_levels` argument removed (it has no meaning without a quantiser).
     Every other argument (`--bpc_supervision`, `--bpc_type`, `--max_steps`,
     `--lr`, `--fine_weight`, `--output_dir`, `--seed`) is kept with identical
     semantics and defaults, except that the default `--output_dir` is
     CHECKPOINTS_DIR/identity_baseline (identity_baseline_bpc with
     --bpc_supervision).
  6. Checkpoint dicts drop `fsq_levels` / `num_codes` and gain
     `quantiser: 'identity'` and `hidden_dim`, so evaluate_fsq.py can identify
     the architecture from the file.
  7. Checkpoint dicts also carry a configuration fingerprint
     (`config_fingerprint`, from models/checkpoint_fingerprint.py), which
     train_fsq.py does not write.

Everything else is the same as in train_fsq.py, including
`LibriSpeechPhonemeDataset(..., 15.0, 1.0)`: min_audio_length = 1.0, as in the
FSQ/RVQ trainers and not the 2.0 used by train_fusion_phoneme_supervised.py
(the dataset class stores this argument but does not use it).
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
import argparse

from paths import ALIGNMENTS_DIR, CHECKPOINTS_DIR, MANIFEST_TRAIN_CLEAN_100, MANIFEST_DEV_CLEAN

from data.phoneme_vocab import PHONEME_TO_ID, normalize_phoneme, VOCAB_SIZE
from data.dataset import reanchor_audio_path
from models.checkpoint_fingerprint import build_fingerprint
from scripts.broad_phonetic_classes import (
    build_phoneme_to_bpc_table,
    IDX_TO_BPC_8,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# === DATA PIPELINE (identical to train_fsq.py) ===

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

class ProtoSpeechIdentity(nn.Module):
    """ProtoSpeech with no quantiser: the identity bottleneck.

    Architecture: CNN encoder -> Transformer -> LayerNorm -> classification head(s)

    The encoder code is the same as that of ProtoSpeechFSQ in scripts/train_fsq.py;
    the only difference is that the 512-d Transformer output goes straight to
    the heads instead of through FSQ. Head shapes are unchanged, so the
    parameter count differs from an FSQ model by exactly the FSQ projections
    (project_down: 512xd, project_up: dx512).
    """

    def __init__(self, hidden_dim: int = 512, num_transformer_layers: int = 6,
                 num_heads: int = 8, num_bpc_classes: int = 8,
                 num_fine_classes: int = 43, bpc_supervision: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        self.bpc_supervision = bpc_supervision
        self.num_bpc_classes = num_bpc_classes
        self.num_fine_classes = num_fine_classes
        self.hidden_dim = hidden_dim

        # CNN Encoder (as in train_fsq.py)
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

        # Transformer (as in train_fsq.py)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            )
            for _ in range(num_transformer_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # No quantiser here: this is the only difference from ProtoSpeechFSQ.

        # Classification heads (identical widths to train_fsq.py)
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
        logger.info(f"ProtoSpeechIdentity: {params/1e6:.2f}M parameters")
        logger.info(f"  Quantiser: none (identity, {hidden_dim}-d continuous)")
        logger.info(f"  BPC supervision: {bpc_supervision}")

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)

        # train_fsq.py passes fsq_out['quantized'] here; this model passes `hidden`.
        result = {'fine_logits': self.fine_head(hidden)}

        if self.bpc_supervision:
            result['bpc_logits'] = self.bpc_head(hidden)

        return result


def main():
    parser = argparse.ArgumentParser(
        description='Train the ProtoSpeech identity baseline (no quantiser)')
    parser.add_argument('--bpc_supervision', action='store_true',
                        help='Enable dual-head BPC + fine supervision')
    parser.add_argument('--bpc_type', type=str, default='8class',
                        choices=['8class'])
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fine_weight', type=float, default=0.5,
                        help='Weight for fine-grained phoneme loss when BPC is active')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint output dir. Default: CHECKPOINTS_DIR/identity_baseline (identity_baseline_bpc with --bpc_supervision).')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (Python/NumPy/PyTorch). '
                             'When set, also forces cudnn deterministic mode.')
    args = parser.parse_args()

    # Seed block: identical to scripts/train_fsq.py, so the determinism
    # settings are the same in every run.
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
    num_bpc = 8
    idx_to_name = IDX_TO_BPC_8
    phoneme_to_bpc = build_phoneme_to_bpc_table(args.bpc_type)
    phoneme_to_bpc_tensor = torch.tensor(phoneme_to_bpc, dtype=torch.long, device=device)

    logger.info("=" * 60)
    logger.info("IDENTITY BASELINE TRAINING (no quantiser)")
    logger.info("=" * 60)
    logger.info("Bottleneck: none (512-d Transformer output -> classifier)")
    logger.info(f"BPC supervision: {args.bpc_supervision}")
    if args.bpc_supervision:
        logger.info(f"BPC classes: {num_bpc} -> {list(idx_to_name.values())}")
        logger.info(f"Fine weight: {args.fine_weight}")
    logger.info(f"VOCAB_SIZE (fine): {VOCAB_SIZE}")
    logger.info("=" * 60)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = "identity_baseline"
        if args.bpc_supervision:
            suffix += "_bpc"
        output_dir = CHECKPOINTS_DIR / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    model = ProtoSpeechIdentity(
        num_bpc_classes=num_bpc,
        num_fine_classes=VOCAB_SIZE,
        bpc_supervision=args.bpc_supervision,
    ).to(device)

    # Record the configuration that was built. preserve_symmetry,
    # extract_layers and similar options change what the weights mean without
    # changing their shape, so load_state_dict cannot detect a mismatch; the
    # loader rebuilds the model from this record and checks it against it.
    FINGERPRINT = build_fingerprint(model, args, num_bpc_classes=num_bpc,
                                    hidden_dim=model.hidden_dim)

    # Data: as in train_fsq.py (min_audio_length = 1.0)
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

            # No vq_loss term: there is no quantiser.

            if torch.isnan(total_loss):
                continue

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Accuracy logging every 200 steps (no Perp/Codes fields, as there is no codebook).
            if global_step % 200 == 0:
                with torch.no_grad():
                    valid_fine = fine_labels_trimmed >= 0
                    fine_preds = fine_logits.argmax(dim=-1)
                    fine_acc = (fine_preds[valid_fine] == fine_labels_trimmed[valid_fine]).float().mean().item()

                    log_msg = (
                        f"Step {global_step} | "
                        f"Fine CE: {fine_loss.item():.3f} | "
                        f"Fine Acc: {fine_acc:.1%}"
                    )

                    if args.bpc_supervision:
                        bpc_preds = bpc_logits.argmax(dim=-1)
                        bpc_acc = (bpc_preds[valid_bpc] == bpc_labels[valid_bpc]).float().mean().item()
                        log_msg = (
                            f"Step {global_step} | "
                            f"BPC CE: {bpc_loss.item():.3f} | Fine CE: {fine_loss.item():.3f} | "
                            f"BPC Acc: {bpc_acc:.1%} | Fine Acc: {fine_acc:.1%}"
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
                        'config_fingerprint': FINGERPRINT,
                        'step': global_step,
                        'fine_acc': val_fine_acc,
                        'quantiser': 'identity',
                        'hidden_dim': model.hidden_dim,
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

    # Final save
    save_dict = {
        'model_state_dict': model.state_dict(),
        'config_fingerprint': FINGERPRINT,
        'step': global_step,
        'best_fine_acc': best_fine_acc,
        'quantiser': 'identity',
        'hidden_dim': model.hidden_dim,
        'bpc_supervision': args.bpc_supervision,
    }
    if args.bpc_supervision:
        save_dict['best_bpc_acc'] = best_bpc_acc
    torch.save(save_dict, output_dir / 'final.pt')

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Quantiser: none (identity baseline)")
    logger.info(f"Best Fine Acc: {best_fine_acc:.2%}")
    if args.bpc_supervision:
        logger.info(f"Best BPC Acc: {best_bpc_acc:.2%}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
