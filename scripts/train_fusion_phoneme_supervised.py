#!/usr/bin/env python3
"""
Train early fusion with direct phoneme supervision.

The concatenated Transformer layers are trained directly on phoneme labels from
TextGrid alignments, with no teacher model, so that the training objective
matches the evaluation metric.
"""
import os
import re
import json
import random
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

import numpy as np

from paths import ALIGNMENTS_DIR, CHECKPOINTS_DIR, MANIFEST_TRAIN_CLEAN_100, MANIFEST_DEV_CLEAN

from data.phoneme_vocab import PHONEME_TO_ID, normalize_phoneme, VOCAB_SIZE
from data.dataset import reanchor_audio_path
from models.checkpoint_fingerprint import build_fingerprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

ALIGNMENT_BASE = str(ALIGNMENTS_DIR)


def parse_textgrid(textgrid_path: str):
    """Parse TextGrid file to get phoneme intervals."""
    with open(textgrid_path, 'r') as f:
        content = f.read()

    phones_match = re.search(
        r'name = "phones".*?intervals: size = (\d+)(.*?)(?=item \[|$)',
        content, re.DOTALL
    )

    if not phones_match:
        return []

    intervals_text = phones_match.group(2)
    intervals = []
    pattern = r'xmin = ([\d.]+)\s+xmax = ([\d.]+)\s+text = "([^"]*)"'

    for match in re.finditer(pattern, intervals_text):
        xmin = float(match.group(1))
        xmax = float(match.group(2))
        phone = match.group(3)
        intervals.append((xmin, xmax, phone))

    return intervals


def get_frame_labels(intervals, num_frames, frame_rate=50):
    """Convert phoneme intervals to frame-level labels."""
    frame_duration = 1.0 / frame_rate
    labels = []

    for frame_idx in range(num_frames):
        frame_center = (frame_idx + 0.5) * frame_duration

        phoneme = '<sil>'
        for xmin, xmax, phone in intervals:
            if xmin <= frame_center < xmax:
                normalized = normalize_phoneme(phone)
                if normalized in PHONEME_TO_ID:
                    phoneme = normalized
                break

        labels.append(PHONEME_TO_ID.get(phoneme, PHONEME_TO_ID.get('<unk>', 0)))

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


class ProtoSpeechPhonemeSupervised(nn.Module):
    """
    Early Fusion with direct phoneme classification head.
    No teacher and no prototypes: the model classifies phonemes directly.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_phonemes: int = 43,
        extract_layers: list = [2, 4, 6],
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extract_layers = extract_layers
        self.fusion_dim = hidden_dim * len(extract_layers)  # 1536

        # CNN Feature Encoder
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(512, 512),
            nn.GELU(),
            nn.Conv1d(512, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Fusion: concatenate layers, layer norm only
        self.fusion_norm = nn.LayerNorm(self.fusion_dim)

        # Direct phoneme classification head
        self.phoneme_head = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_phonemes),
        )

        logger.info(f"ProtoSpeechPhonemeSupervised:")
        logger.info(f"  Extract layers: {extract_layers}")
        logger.info(f"  Fusion dim: {self.fusion_dim}")
        logger.info(f"  Phoneme classes: {num_phonemes}")

    def forward(self, waveform: torch.Tensor):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Ensure minimum length
        min_len = 3200
        if waveform.size(-1) < min_len:
            waveform = F.pad(waveform, (0, min_len - waveform.size(-1)))

        cnn_out = self.feature_encoder(waveform)
        hidden = cnn_out.transpose(1, 2)

        layer_outputs = {}
        for layer_idx, transformer_layer in enumerate(self.transformer_layers, start=1):
            hidden = transformer_layer(hidden)
            if layer_idx in self.extract_layers:
                layer_outputs[f'hidden_L{layer_idx}'] = hidden

        hidden = self.layer_norm(hidden)

        # Concatenate extracted layers
        concat_hidden = torch.cat(
            [layer_outputs[f'hidden_L{idx}'] for idx in self.extract_layers],
            dim=-1
        )  # (B, T, 1536)

        # Fusion (just layer norm)
        fused = self.fusion_norm(concat_hidden)

        # Phoneme classification
        logits = self.phoneme_head(fused)

        return {
            'hidden_states': fused,
            'logits': logits,
            **layer_outputs,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--extract_layers', type=str, default='2,4,6',
                        help='Comma-separated transformer layers to concatenate. '
                             'Use 4,5,6 to match the Gated baseline.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Checkpoint output dir. Default: CHECKPOINTS_DIR/phoneme_supervised.')
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

    extract_layers = [int(x) for x in args.extract_layers.split(',')]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_phonemes = VOCAB_SIZE  # 43

    logger.info("="*60)
    logger.info("DIRECT PHONEME SUPERVISION (No Teacher)")
    logger.info("="*60)
    logger.info("Training directly on phoneme labels from TextGrid")
    logger.info(f"Phoneme classes: {num_phonemes}")
    logger.info("="*60)

    output_dir = (Path(args.output_dir) if args.output_dir
                  else CHECKPOINTS_DIR / 'phoneme_supervised')
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    # Model
    model = ProtoSpeechPhonemeSupervised(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        num_phonemes=num_phonemes,
        extract_layers=extract_layers,
        dropout=0.1,
    ).to(device)

    # Record the configuration that was built. preserve_symmetry,
    # extract_layers and similar options change what the weights mean without
    # changing their shape, so load_state_dict cannot detect a mismatch; the
    # loader rebuilds the model from this record and checks it against it.
    FINGERPRINT = build_fingerprint(model, args, extract_layers=extract_layers, hidden_dim=512)

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Data with phoneme labels
    train_loader = DataLoader(
        LibriSpeechPhonemeDataset(
            str(MANIFEST_TRAIN_CLEAN_100),
            max_audio_length=15.0, min_audio_length=2.0
        ),
        batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        LibriSpeechPhonemeDataset(
            str(MANIFEST_DEV_CLEAN),
            max_audio_length=15.0, min_audio_length=2.0
        ),
        batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.max_steps, pct_start=0.08)

    global_step = 0
    best_val_acc = 0.0

    logger.info("Starting training...")

    for epoch in range(10000):
        model.train()

        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            waveform = batch['waveform'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(waveform)
            logits = outputs['logits']

            # Align lengths
            min_len = min(logits.size(1), labels.size(1))
            logits = logits[:, :min_len]
            labels = labels[:, :min_len]

            # Cross-entropy with ignore_index=-100 for padding
            loss = F.cross_entropy(
                logits.reshape(-1, num_phonemes),
                labels.reshape(-1),
                ignore_index=-100
            )

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 100 == 0:
                with torch.no_grad():
                    valid_mask = labels >= 0
                    if valid_mask.sum() > 0:
                        preds = logits.argmax(dim=-1)
                        acc = (preds[valid_mask] == labels[valid_mask]).float().mean().item()
                        unique = preds[valid_mask].unique().numel()
                    else:
                        acc, unique = 0, 0

                logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | "
                            f"Acc: {acc:.1%} | Unique: {unique}/{num_phonemes}")

            if global_step % 2000 == 0 and global_step > 0:
                model.eval()
                val_correct, val_total = 0, 0

                with torch.no_grad():
                    for vb in val_loader:
                        vw = vb['waveform'].to(device)
                        vl = vb['labels'].to(device)

                        out = model(vw)
                        vlogits = out['logits']

                        ml = min(vlogits.size(1), vl.size(1))
                        vlogits = vlogits[:, :ml]
                        vl = vl[:, :ml]

                        valid = vl >= 0
                        vpreds = vlogits.argmax(dim=-1)

                        val_correct += (vpreds[valid] == vl[valid]).sum().item()
                        val_total += valid.sum().item()

                avg_val = val_correct / val_total if val_total > 0 else 0
                logger.info(f"[VAL] Step {global_step} | Acc: {avg_val:.1%}")

                if avg_val > best_val_acc:
                    best_val_acc = avg_val
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'config_fingerprint': FINGERPRINT,
                        'step': global_step,
                        'val_acc': avg_val,
                    }, output_dir / 'best.pt')
                    logger.info(f"Saved best model (acc={avg_val:.1%})")

                model.train()

            global_step += 1

        if global_step >= args.max_steps:
            break

    torch.save({
        'model_state_dict': model.state_dict(),
        'config_fingerprint': FINGERPRINT,
        'step': global_step,
        'best_val_acc': best_val_acc,
    }, output_dir / 'final.pt')

    logger.info("="*60)
    logger.info(f"Training complete! Best val acc: {best_val_acc:.1%}")
    logger.info("="*60)
    logger.info("This is the phoneme accuracy (no probe needed)")


if __name__ == '__main__':
    main()
