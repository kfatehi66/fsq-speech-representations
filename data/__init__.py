"""Data loading utilities for ProtoSpeech."""

from .dataset import LibriSpeechDataset, collate_fn

__all__ = ['LibriSpeechDataset', 'collate_fn']
