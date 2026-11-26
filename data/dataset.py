"""
LibriSpeech Dataset for ProtoSpeech training.

Loads audio from JSON manifest files and handles:
- Variable length audio with padding
- Length filtering (min/max duration)
- Random cropping for long audio
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

from paths import LIBRISPEECH_AUDIO, MANIFEST_TRAIN_CLEAN_100, MANIFEST_AUDIO_PREFIX

# Matches MANIFEST_AUDIO_PREFIX (env var PROTOSPEECH_MANIFEST_AUDIO_PREFIX) at
# the start of an audio path. With the prefix unset the pattern is "^", whose
# substitution leaves every path unchanged.
_MANIFEST_AUDIO_PREFIX_RE = re.compile(
    "^" + re.escape(MANIFEST_AUDIO_PREFIX)
)


def reanchor_audio_path(audio_path: str) -> str:
    """Re-anchor a manifest audio_path at LIBRISPEECH_AUDIO/LibriSpeech.

    Idempotent across the manifest formats the codebase reads, provided that
    LIBRISPEECH_AUDIO is absolute and, if MANIFEST_AUDIO_PREFIX is non-empty,
    that no re-anchored path starts with it:
      * Absolute under MANIFEST_AUDIO_PREFIX:  <prefix>split/x/y/z.flac
        -> regex strips prefix -> "split/x/y/z.flac" -> joined under audio_root.
      * Relative:  "split/x/y/z.flac"
        -> regex no-op -> joined under audio_root.
      * Already-resolved absolute:  /path/to/LibriSpeech/split/x/y/z.flac
        -> regex no-op; Path("/a") / "/b" yields "/b" -> input wins, unchanged.
      * Any other absolute path -> same passthrough as case 3.
    """
    audio_root = LIBRISPEECH_AUDIO / "LibriSpeech"
    stripped = _MANIFEST_AUDIO_PREFIX_RE.sub("", audio_path)
    return str(audio_root / stripped)


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset for self-supervised speech learning.

    Args:
        manifest_path: Path to JSON manifest file
        sample_rate: Target sample rate (default: 16000)
        max_audio_length: Maximum audio length in seconds (default: 10.0)
        min_audio_length: Minimum audio length in seconds (default: 1.0)
        random_crop: If True, randomly crop long audio; else truncate
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        max_audio_length: float = 10.0,
        min_audio_length: float = 1.0,
        random_crop: bool = True,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.min_samples = int(min_audio_length * sample_rate)
        self.random_crop = random_crop

        # Load manifest
        self.samples = self._load_manifest(manifest_path)
        print(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load and filter samples from manifest."""
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        filtered = []
        for sample in data:
            sample['audio_path'] = reanchor_audio_path(sample['audio_path'])
            if 'duration' in sample:
                duration_samples = int(sample['duration'] * self.sample_rate)
                if duration_samples < self.min_samples:
                    continue
            filtered.append(sample)

        return filtered

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        audio_path = sample['audio_path']

        # Load audio
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return a random other sample
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # (T,)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Handle length
        if waveform.shape[0] > self.max_samples:
            if self.random_crop:
                # Random crop
                start = random.randint(0, waveform.shape[0] - self.max_samples)
                waveform = waveform[start:start + self.max_samples]
            else:
                # Truncate from start
                waveform = waveform[:self.max_samples]

        # Skip too short (shouldn't happen after filtering, but safety check)
        if waveform.shape[0] < self.min_samples:
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            'waveform': waveform,
            'length': waveform.shape[0],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.

    Pads waveforms to same length within batch.

    Returns:
        Dictionary with:
            - waveform: (B, T) padded waveforms
            - lengths: (B,) original lengths
            - padding_mask: (B, T) True where padded
    """
    waveforms = [item['waveform'] for item in batch]
    lengths = [item['length'] for item in batch]

    # Pad to max length in batch
    max_len = max(lengths)

    padded_waveforms = []
    padding_masks = []

    for waveform, length in zip(waveforms, lengths):
        if length < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - length)
            padded = torch.cat([waveform, padding])
            mask = torch.cat([
                torch.zeros(length, dtype=torch.bool),
                torch.ones(max_len - length, dtype=torch.bool)
            ])
        else:
            padded = waveform
            mask = torch.zeros(length, dtype=torch.bool)

        padded_waveforms.append(padded)
        padding_masks.append(mask)

    return {
        'waveform': torch.stack(padded_waveforms),
        'lengths': torch.tensor(lengths),
        'padding_mask': torch.stack(padding_masks),
    }


def create_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int = 8,
    num_workers: int = 4,
    sample_rate: int = 16000,
    max_audio_length: float = 10.0,
    min_audio_length: float = 1.0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_manifest: Path to training manifest
        val_manifest: Path to validation manifest
        batch_size: Batch size
        num_workers: Number of dataloader workers
        sample_rate: Audio sample rate
        max_audio_length: Maximum audio length in seconds
        min_audio_length: Minimum audio length in seconds
        pin_memory: Pin memory for GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = LibriSpeechDataset(
        manifest_path=train_manifest,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
        min_audio_length=min_audio_length,
        random_crop=True,
    )

    val_dataset = LibriSpeechDataset(
        manifest_path=val_manifest,
        sample_rate=sample_rate,
        max_audio_length=max_audio_length,
        min_audio_length=min_audio_length,
        random_crop=False,  # Deterministic for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    manifest = str(MANIFEST_TRAIN_CLEAN_100)

    dataset = LibriSpeechDataset(manifest, max_audio_length=10.0)
    print(f"Dataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"Sample waveform shape: {sample['waveform'].shape}")
    print(f"Sample length: {sample['length']}")

    # Test dataloader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\nBatch waveform shape: {batch['waveform'].shape}")
    print(f"Batch lengths: {batch['lengths']}")
    print(f"Batch padding_mask shape: {batch['padding_mask'].shape}")
