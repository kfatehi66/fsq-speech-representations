#!/usr/bin/env python3
"""
Evaluation script for FSQ experiments.

Produces:
1. Per-phoneme accuracy (the 39 ARPABET phonemes and <sil>, where labelled frames exist)
2. Per-BPC-class accuracy (the classes of BPC_ORDER that have labelled frames)
3. BPC-level confusion matrix heatmap (.png)
4. Error budget plot: error contribution % vs class frequency % (.png)
5. Code usage histogram (.png)
6. Summary JSON with all metrics

With --checkpoint only the summary JSON is written; the plots 3 to 5 come from
the multi-model mode that runs without --checkpoint.

Can also evaluate RVQ baselines for direct comparison.
"""

import os
import sys
import re
import json
import math
import random
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
from collections import defaultdict, OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from paths import ALIGNMENTS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, MANIFEST_DEV_CLEAN
from data.dataset import reanchor_audio_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ================================================================
# PATHS
# ================================================================
CHECKPOINT_BASE = CHECKPOINTS_DIR
OUTPUT_DIR = OUTPUTS_DIR / 'fsq'
ALIGNMENT_BASE = str(ALIGNMENTS_DIR)
VAL_MANIFEST = str(MANIFEST_DEV_CLEAN)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# PHONEME VOCABULARY
# ================================================================
CMU_PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY',
    'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH',
]
SPECIAL_TOKENS = ['<pad>', '<blank>', '<unk>', '<sil>']
PHONEME_VOCAB = SPECIAL_TOKENS + CMU_PHONEMES
PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
ID_TO_PHONEME = {i: p for i, p in enumerate(PHONEME_VOCAB)}
VOCAB_SIZE = len(PHONEME_VOCAB)  # 43


def normalize_phoneme(phoneme):
    phoneme = phoneme.rstrip('012')
    if phoneme in ('', 'sp', 'sil', 'SIL', 'spn', 'SPN'):
        return '<sil>'
    phoneme = phoneme.upper()
    if phoneme in PHONEME_TO_ID:
        return phoneme
    return '<unk>'


# ================================================================
# BPC MAPPINGS
# ================================================================
BPC_8_MAP = {
    'CH': 'Affricate', 'JH': 'Affricate',
    'AW': 'Diphthong', 'AY': 'Diphthong', 'EY': 'Diphthong',
    'OW': 'Diphthong', 'OY': 'Diphthong',
    'DH': 'Fricative', 'F': 'Fricative', 'S': 'Fricative',
    'SH': 'Fricative', 'TH': 'Fricative', 'V': 'Fricative',
    'Z': 'Fricative', 'ZH': 'Fricative',
    'M': 'Nasal', 'N': 'Nasal', 'NG': 'Nasal',
    'B': 'Plosive', 'D': 'Plosive', 'G': 'Plosive',
    'K': 'Plosive', 'P': 'Plosive', 'T': 'Plosive',
    'HH': 'Semi-vowel', 'L': 'Semi-vowel', 'R': 'Semi-vowel',
    'W': 'Semi-vowel', 'Y': 'Semi-vowel',
    'AA': 'Vowel', 'AE': 'Vowel', 'AH': 'Vowel', 'AO': 'Vowel',
    'EH': 'Vowel', 'ER': 'Vowel', 'IH': 'Vowel', 'IY': 'Vowel',
    'UH': 'Vowel', 'UW': 'Vowel',
    '<sil>': 'Silence',
}
BPC_ORDER = ['Affricate', 'Diphthong', 'Fricative', 'Nasal',
             'Plosive', 'Semi-vowel', 'Vowel', 'Silence']
BPC_TO_IDX = {name: i for i, name in enumerate(BPC_ORDER)}

PID_TO_BPC = {}
for pid, pname in ID_TO_PHONEME.items():
    if pname in BPC_8_MAP:
        PID_TO_BPC[pid] = BPC_8_MAP[pname]

BPC_MEMBERS = defaultdict(list)
for pname, bpc in BPC_8_MAP.items():
    if pname in PHONEME_TO_ID:
        BPC_MEMBERS[bpc].append(pname)


# ================================================================
# DATA PIPELINE
# ================================================================
def parse_textgrid(textgrid_path):
    intervals = []
    with open(textgrid_path, 'r') as f:
        content = f.read()
    phone_match = re.search(
        r'name = "phones".*?intervals: size = (\d+)(.*?)(?:item \[|$)',
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


def get_textgrid_path(audio_path):
    parts = audio_path.split('/')
    split = parts[-4]
    speaker_id = parts[-3]
    chapter_id = parts[-2]
    filename = parts[-1].replace('.flac', '.TextGrid').replace('.wav', '.TextGrid')
    return f"{ALIGNMENT_BASE}/{split}/{speaker_id}/{chapter_id}/{filename}"


class LibriSpeechPhonemeDataset(Dataset):
    def __init__(self, manifest_path, max_audio_length=30.0,
                 min_audio_length=1.0, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        with open(manifest_path, 'r') as f:
            data = json.load(f)
        self.samples = []
        for item in data:
            item['audio_path'] = reanchor_audio_path(item['audio_path'])
            tg = get_textgrid_path(item['audio_path'])
            if os.path.exists(tg):
                self.samples.append(item)
        logger.info(f"Loaded {len(self.samples)} samples from {manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform, sr = torchaudio.load(item['audio_path'])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
        if waveform.shape[0] > self.max_samples:
            waveform = waveform[:self.max_samples]
        intervals = parse_textgrid(get_textgrid_path(item['audio_path']))
        num_frames = waveform.shape[0] // 320
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
    max_wav = max(w.shape[0] for w in waveforms)
    max_lab = max(l.shape[0] for l in labels)
    pw, pl = [], []
    for w, l in zip(waveforms, labels):
        if w.shape[0] < max_wav:
            w = torch.cat([w, torch.zeros(max_wav - w.shape[0])])
        pw.append(w)
        if l.shape[0] < max_lab:
            l = torch.cat([l, torch.full((max_lab - l.shape[0],), -100, dtype=torch.long)])
        pl.append(l)
    return {
        'waveform': torch.stack(pw),
        'labels': torch.stack(pl),
        'lengths': torch.tensor(lengths),
    }


# ================================================================
# MODEL DEFINITIONS (self-contained for inference)
# ================================================================

# --- FSQ ---
# ProtoSpeechFSQ, ProtoSpeechRVQ and ProtoSpeechBPC below are hand-copied
# inference clones of their trainers, and no load path in this file uses them:
# both the single-checkpoint path and the legacy MODEL_CONFIGS path go through
# models/checkpoint_loading.py, which builds the trainer's own class.
# ProtoSpeechFSQ is kept because tests/test_fsq_e2e_regression.py imports it
# by name; no other file of this repository imports these classes.
#
# The quantiser itself is not cloned: models/fsq.py is imported, so evaluation
# and training use the same quantiser definition. A copy could diverge from the
# training-time quantiser whenever either is edited.
from models.fsq import FSQ  # noqa: E402
from models.checkpoint_loading import load_model_from_checkpoint  # noqa: E402
from scripts.rate_analysis import (  # noqa: E402
    FSQRateAccumulator, RVQRateAccumulator,
)


def _fsq_out_to_eval_dict(fsq_out):
    """Adapt models.fsq.FSQ's output dict to the flat keys used below.

    models.fsq.FSQ returns list-wrapped metrics to match the ResidualVQ
    interface; this eval path wants the single FSQ "layer" unwrapped.
    """
    return {
        'quantized': fsq_out['quantized'],
        'indices': fsq_out['all_indices'][0],
        'codes': fsq_out['codes'],
        'perplexity': fsq_out['perplexities'][0],
        'unique_codes': fsq_out['unique_codes'][0],
    }


class ProtoSpeechFSQ(nn.Module):
    def __init__(self, fsq_levels=None, num_bpc_classes=8,
                 num_fine_classes=43, bpc_supervision=False,
                 hidden_dim=512, dropout=0.1):
        super().__init__()
        if fsq_levels is None:
            fsq_levels = [8, 5, 5, 5]
        self.bpc_supervision = bpc_supervision

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
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(6)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fsq = FSQ(input_dim=hidden_dim, levels=fsq_levels)
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

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)
        fsq_out = _fsq_out_to_eval_dict(self.fsq(hidden))
        out = {
            'fine_logits': self.fine_head(fsq_out['quantized']),
            'indices': fsq_out['indices'],
            'codes': fsq_out['codes'],
            'perplexity': fsq_out['perplexity'],
            'unique_codes': fsq_out['unique_codes'],
        }
        if self.bpc_supervision:
            out['bpc_logits'] = self.bpc_head(fsq_out['quantized'])
        return out


# --- RVQ (for baseline comparison) ---
class VQLayer(nn.Module):
    def __init__(self, num_codes, code_dim):
        super().__init__()
        self.num_codes = num_codes
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

    def forward(self, z):
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
        z_q_st = z + (z_q - z).detach()

        flat = indices.reshape(-1)
        encodings = F.one_hot(flat, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {'quantized': z_q_st, 'indices': indices, 'perplexity': perplexity}


class ResidualVQ(nn.Module):
    def __init__(self, num_layers, num_codes=128, code_dim=512):
        super().__init__()
        self.num_codes = num_codes
        self.vq_layers = nn.ModuleList([
            VQLayer(num_codes, code_dim) for _ in range(num_layers)
        ])

    def forward(self, z):
        quantized_sum = torch.zeros_like(z)
        residual = z
        all_indices = []
        all_perplexities = []
        for layer in self.vq_layers:
            out = layer(residual)
            quantized_sum = quantized_sum + out['quantized']
            residual = z - quantized_sum
            all_indices.append(out['indices'])
            all_perplexities.append(out['perplexity'])
        return {
            'quantized': quantized_sum,
            'all_indices': all_indices,
            'perplexities': all_perplexities,
        }


class ProtoSpeechRVQ(nn.Module):
    def __init__(self, num_vq_layers=4, num_codes=128, hidden_dim=512, dropout=0.1):
        super().__init__()
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
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(6)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.rvq = ResidualVQ(num_vq_layers, num_codes, hidden_dim)
        self.phoneme_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, VOCAB_SIZE)
        )

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)
        rvq_out = self.rvq(hidden)
        logits = self.phoneme_head(rvq_out['quantized'])
        return {
            'fine_logits': logits,
            'all_indices': rvq_out['all_indices'],
            'perplexities': rvq_out['perplexities'],
        }


class ProtoSpeechBPC(nn.Module):
    def __init__(self, num_vq_layers=4, num_codes=128, num_bpc_classes=8,
                 hidden_dim=512, dropout=0.1):
        super().__init__()
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
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=8,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(6)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.rvq = ResidualVQ(num_vq_layers, num_codes, hidden_dim)
        self.bpc_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bpc_classes)
        )
        self.fine_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, VOCAB_SIZE)
        )

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        hidden = self.feature_encoder(waveform).transpose(1, 2)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        hidden = self.layer_norm(hidden)
        rvq_out = self.rvq(hidden)
        q = rvq_out['quantized']
        return {
            'fine_logits': self.fine_head(q),
            'bpc_logits': self.bpc_head(q),
            'all_indices': rvq_out['all_indices'],
            'perplexities': rvq_out['perplexities'],
        }


# ================================================================
# TRAINER MODELS -> THE KEYS THIS EVALUATOR READS
# ================================================================
# Every model is built by models/checkpoint_loading.py, which constructs the
# trainer's own class, so there is no separate inference copy that could drift.
# The trainers' forward() dicts do not agree with each other, because each was
# written for its own training loop:
#
#   train_rvq_supervised.ProtoSpeechRVQ  calls its phoneme output 'logits' and
#       does not return the per-layer code indices at all (its ResidualVQ
#       simply never collects them);
#   train_fsq.ProtoSpeechFSQ             returns list-wrapped 'all_indices' /
#       'perplexities', where the code below reads the unwrapped 'indices';
#   train_gated_supervised, train_fusion_phoneme_supervised
#                                        also say 'logits'.
#
# The adapter below normalises the keys at the one point that every model
# passes. The missing per-layer indices are recovered with forward hooks, which
# read what the model already computes and change nothing.

class _EvalOutputAdapter(nn.Module):
    """Present a trainer model's forward() under this file's key names.

    Wrapping happens after the checkpoint is loaded, so the wrapper never
    appears in any state_dict key.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._layer_indices = []
        self._handles = []
        for m in inner.modules():
            # VQLayer is the only module that owns per-layer code indices. Match
            # by name rather than by class object: each trainer declares its own
            # copy, so there is no single class to compare against.
            if type(m).__name__ == 'VQLayer':
                self._handles.append(m.register_forward_hook(self._vq_hook))

    def _vq_hook(self, module, inputs, output):
        if isinstance(output, dict) and output.get('indices') is not None:
            self._layer_indices.append(output['indices'])

    def forward(self, waveform):
        self._layer_indices = []
        out = dict(self.inner(waveform))

        if 'fine_logits' not in out and 'logits' in out:
            out['fine_logits'] = out['logits']

        # RVQ layers in forward order, for the model whose forward() drops them.
        if 'all_indices' not in out and self._layer_indices:
            out['all_indices'] = list(self._layer_indices)

        # FSQ is a single "layer"; unwrap it so the code-histogram path sees the
        # flat index tensor it expects. FSQ returns [None] once the codebook
        # exceeds int64 (d >= 40), and None is the signal to fall through to the
        # per-dimension `codes`, so it is propagated rather than dropped.
        ai = out.get('all_indices')
        if 'indices' not in out and isinstance(ai, (list, tuple)) and len(ai) == 1:
            out['indices'] = ai[0]

        return out


def _load_eval_model(path, model_type, *, device, overrides=None, **kwargs):
    """The one load path: build from the fingerprint, verify, load, adapt.

    Returns (model, info). `info['config_used']` is what the model was actually
    built with, which is what the rate accumulators are sized from. Sizing them
    from the CLI instead could measure a codebook that the checkpoint does not
    have.
    """
    model, info = load_model_from_checkpoint(
        path, model_type=model_type, overrides=overrides, device=device,
        strict_class=True, log=logger, **kwargs)
    return _EvalOutputAdapter(model).to(device).eval(), info


# Model types with no discrete codes at all. rate_analysis is emitted as null
# with a reason rather than omitted, so a consumer can tell "not applicable"
# apart from "this eval predates the instrumentation".
_UNQUANTISED_TYPES = {
    'identity': 'Identity baseline: no quantiser, the 512-d Transformer output '
                'goes straight to the classifier. There are no discrete codes, '
                'so rate, entropy and utilisation are undefined. This run is '
                'the rate=infinity end of the rate-accuracy curve.',
    'gated': 'Gated fusion baseline (no VQ): a softmax-gated combination of '
             'Transformer layers, continuous throughout. No discrete codes.',
    'fusion': 'Concat fusion baseline (no VQ): layers concatenated to 1536-d '
              'and layer-normed, continuous throughout. No discrete codes.',
}


# ================================================================
# EVALUATION
# ================================================================
def _log_rate_analysis(ra):
    """Human-readable echo of the rate_analysis block."""
    logger.info("=" * 66)
    logger.info("RATE ANALYSIS (%s)  n_frames=%s", ra['quantiser'], f"{ra['n_frames']:,}")
    if ra['quantiser'] == 'fsq':
        logger.info("  levels=%s  d=%d", ra['levels'], ra['d'])
        logger.info("  sum_i H(dim_i)      : %8.3f bits  (nominal %.3f, eff %.1f%%)",
                    ra['marginal_entropy_bits_sum'], ra['nominal_rate_bits'],
                    ra['rate_efficiency'] * 100)
        used = [x['n_used_levels'] for x in ra['per_dimension']]
        logger.info("  levels used per dim : %s", used)
    else:
        perp = ", ".join(f"{p:.1f}" for p in ra['per_layer_perplexity'])
        used = [x['n_used_codes'] for x in ra['per_layer']]
        logger.info("  per-layer perplexity: [%s]", perp)
        logger.info("  codes used / %d     : %s", ra['num_codes'], used)
        logger.info("  sum_l H_l           : %8.3f bits  (nominal %.3f, eff %.1f%%)",
                    ra['entropy_rate_bits_sum'], ra['nominal_rate_bits'],
                    ra['rate_efficiency'] * 100)
    logger.info("  joint H (plug-in)   : %8.3f bits   (Miller-Madow %.3f)",
                ra['joint_entropy_bits'], ra['joint_entropy_bits_miller_madow'])
    logger.info("  unique codes        : %s   code id: %s",
                f"{ra['n_unique_codes']:,}", ra['joint_code_id'])
    if ra['estimator_saturated']:
        logger.info("  Warning: joint entropy is sample-limited "
                    "(codebook > n_frames; bound = %.2f bits)",
                    ra['plugin_upper_bound_bits'])
    logger.info("=" * 66)


def evaluate_model(model, loader, device, rate_acc=None, collect_codes=True):
    """Per-phoneme accuracy + confusion matrix + code utilization.

    Args:
        rate_acc: optional FSQRateAccumulator / RVQRateAccumulator. When given,
            it is updated per batch with the streaming histograms used for the
            "rate_analysis" JSON block.
        collect_codes: when False, the raw per-frame code tensors are not
            concatenated. The legacy multi-model path needs them for the code
            histogram plots; the single-checkpoint path does not, and for
            FSQ-64d over a full split the concatenation is prohibitive.
    """
    model.eval()
    correct = np.zeros(VOCAB_SIZE, dtype=np.int64)
    total = np.zeros(VOCAB_SIZE, dtype=np.int64)
    confusion = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.int64)

    all_code_indices = []

    with torch.no_grad():
        for batch in loader:
            wav = batch['waveform'].to(device)
            labels = batch['labels'].to(device)
            out = model(wav)
            logits = out['fine_logits']
            preds = logits.argmax(dim=-1)

            min_len = min(preds.size(1), labels.size(1))
            preds_flat = preds[:, :min_len].reshape(-1).cpu().numpy()
            labels_flat = labels[:, :min_len].reshape(-1).cpu().numpy()

            valid = labels_flat >= 0
            vl = labels_flat[valid]
            vp = preds_flat[valid]

            for c in range(VOCAB_SIZE):
                mask = vl == c
                total[c] += mask.sum()
                correct[c] += (vp[mask] == c).sum()

            np.add.at(confusion.ravel(), vl * VOCAB_SIZE + vp, 1)

            # Streaming rate/entropy histograms (never stores raw frames).
            if rate_acc is not None:
                if isinstance(rate_acc, RVQRateAccumulator):
                    rate_acc.update(out['all_indices'])
                else:
                    rate_acc.update(out['codes'])

            # Collect code indices / codes
            if collect_codes:
                if 'indices' in out and out['indices'] is not None:
                    all_code_indices.append(out['indices'].reshape(-1).cpu())
                elif 'codes' in out and out['codes'] is not None:
                    all_code_indices.append(out['codes'].reshape(-1, out['codes'].shape[-1]).cpu())
                elif 'all_indices' in out:
                    for layer_idx in out['all_indices']:
                        all_code_indices.append(layer_idx.reshape(-1).cpu())

    code_data = torch.cat(all_code_indices) if all_code_indices else None
    return correct, total, confusion, code_data


# ================================================================
# AGGREGATION
# ================================================================
def aggregate_to_bpc(correct, total):
    bpc_c = {b: 0 for b in BPC_ORDER}
    bpc_t = {b: 0 for b in BPC_ORDER}
    for pid in range(VOCAB_SIZE):
        if pid in PID_TO_BPC and total[pid] > 0:
            bpc = PID_TO_BPC[pid]
            bpc_c[bpc] += int(correct[pid])
            bpc_t[bpc] += int(total[pid])
    total_errors = sum(bpc_t[b] - bpc_c[b] for b in BPC_ORDER)
    total_samples = sum(bpc_t[b] for b in BPC_ORDER)
    results = {}
    for bpc in BPC_ORDER:
        if bpc_t[bpc] > 0:
            acc = bpc_c[bpc] / bpc_t[bpc]
            errs = bpc_t[bpc] - bpc_c[bpc]
            results[bpc] = {
                'accuracy': acc,
                'frequency': bpc_t[bpc] / total_samples if total_samples else 0,
                'error_count': int(errs),
                'error_contribution': errs / total_errors if total_errors else 0,
                'total': int(bpc_t[bpc]),
                'correct': int(bpc_c[bpc]),
            }
    return results


def per_phoneme_table(correct, total):
    results = {}
    for pid in range(VOCAB_SIZE):
        pname = ID_TO_PHONEME[pid]
        if pid in PID_TO_BPC and total[pid] > 0:
            results[pname] = {
                'accuracy': correct[pid] / total[pid],
                'total': int(total[pid]),
                'correct': int(correct[pid]),
                'bpc': PID_TO_BPC[pid],
            }
    return results


def aggregate_confusion_bpc(confusion):
    n = len(BPC_ORDER)
    bpc_cm = np.zeros((n, n))
    for ti in range(VOCAB_SIZE):
        for pi in range(VOCAB_SIZE):
            if ti in PID_TO_BPC and pi in PID_TO_BPC:
                bpc_cm[BPC_TO_IDX[PID_TO_BPC[ti]], BPC_TO_IDX[PID_TO_BPC[pi]]] += confusion[ti, pi]
    return bpc_cm


# ================================================================
# PLOTTING
# ================================================================
def plot_error_budget(bpc_results, title, path):
    classes = [b for b in BPC_ORDER if b in bpc_results]
    err_c = [bpc_results[c]['error_contribution'] * 100 for c in classes]
    freq = [bpc_results[c]['frequency'] * 100 for c in classes]

    x = np.arange(len(classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x - w/2, err_c, w, label='Error Contribution %',
                color='coral', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x + w/2, freq, w, label='Class Frequency %',
                color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, f'{h:.1f}',
                ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_confusion_heatmap(cm, labels, title, path):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Predicted %'}, linewidths=0.5)
    ax.set_xlabel('Predicted BPC', fontsize=12)
    ax.set_ylabel('True BPC', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.xticks(rotation=30, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_phoneme_confusion(confusion, title, path):
    """Full 39-phoneme confusion matrix (excluding special tokens)."""
    # Extract CMU phonemes only (ids 4-42)
    cmu_ids = list(range(4, VOCAB_SIZE))
    cmu_names = [ID_TO_PHONEME[i] for i in cmu_ids]
    cm = confusion[np.ix_(cmu_ids, cmu_ids)]

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=cmu_names, yticklabels=cmu_names, ax=ax,
                linewidths=0.3, annot_kws={'size': 6},
                cbar_kws={'label': 'Predicted %'})
    ax.set_xlabel('Predicted Phoneme', fontsize=12)
    ax.set_ylabel('True Phoneme', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


def plot_code_histogram(code_indices, num_codes, title, path):
    """Histogram of code usage across the validation set."""
    total = code_indices.numel()
    avg_freq = total / num_codes

    if num_codes <= 100000:
        counts = torch.zeros(num_codes)
        counts.scatter_add_(0, code_indices, torch.ones_like(code_indices, dtype=torch.float32))
        counts_np = counts.numpy()
        sorted_counts = np.sort(counts_np)[::-1]
        used = (counts_np > 0).sum()
        nonzero = counts_np[counts_np > 0]
    else:
        # Large codebook: use unique() to avoid huge bar charts
        unique_vals, inv_counts = code_indices.unique(return_counts=True)
        used = unique_vals.numel()
        sorted_counts = np.sort(inv_counts.numpy().astype(float))[::-1]
        nonzero = sorted_counts  # all non-zero by construction

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: sorted usage curve (only used codes for large codebooks)
    ax1.bar(range(len(sorted_counts)), sorted_counts, width=1.0,
            color='steelblue', edgecolor='none', alpha=0.8)
    ax1.axhline(y=avg_freq, color='red', linestyle='--', alpha=0.7,
                label=f'Avg frequency ({avg_freq:.1f})')
    if num_codes <= 100000:
        ax1.axhline(y=0.01 * avg_freq, color='orange', linestyle='--', alpha=0.7,
                    label=f'Dead threshold (1% avg)')
    ax1.set_xlabel('Code rank', fontsize=11)
    ax1.set_ylabel('Usage count', fontsize=11)
    ax1.set_title(f'{title}\nUsed: {used}/{num_codes} ({used/num_codes*100:.1f}%)',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_yscale('log')

    # Right: histogram of usage counts
    if len(nonzero) > 1:
        ax2.hist(nonzero, bins=min(50, len(nonzero)), color='steelblue',
                 edgecolor='black', linewidth=0.3, alpha=0.8)
        ax2.set_xlabel('Usage count', fontsize=11)
        ax2.set_ylabel('Number of codes', fontsize=11)
        ax2.set_title(f'Usage distribution (non-zero codes)', fontsize=11)
    elif len(nonzero) == 1:
        ax2.bar([0], nonzero, color='steelblue', edgecolor='black')
        ax2.set_title(f'Single code used ({int(nonzero[0])} times)', fontsize=11)
    else:
        ax2.text(0.5, 0.5, 'No codes used', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {path}")


# ================================================================
# PRINT HELPERS
# ================================================================
def print_bpc_table(bpc_results, model_name):
    print(f"\n  {model_name}")
    print(f"  {'BPC Class':<14} {'Accuracy':>8} {'Freq':>7} {'Errors':>8} {'ErrContrib':>11} {'Samples':>9}")
    print('  ' + '-' * 60)
    for bpc in BPC_ORDER:
        if bpc in bpc_results:
            r = bpc_results[bpc]
            print(f"  {bpc:<14} {r['accuracy']:>7.1%} {r['frequency']:>6.1%} "
                  f"{r['error_count']:>8,} {r['error_contribution']:>10.1%} {r['total']:>9,}")
    total_correct = sum(bpc_results[b]['correct'] for b in BPC_ORDER if b in bpc_results)
    total_samples = sum(bpc_results[b]['total'] for b in BPC_ORDER if b in bpc_results)
    print('  ' + '-' * 60)
    print(f"  {'TOTAL':<14} {total_correct/total_samples:>7.1%} {'100.0%':>7} "
          f"{total_samples - total_correct:>8,} {'100.0%':>11} {total_samples:>9,}")


def print_per_phoneme(phoneme_results, bpc_class=None):
    items = list(phoneme_results.items())
    if bpc_class:
        items = [(p, r) for p, r in items if r['bpc'] == bpc_class]
    items.sort(key=lambda x: x[1]['accuracy'])
    for pname, r in items:
        bar = '#' * int(r['accuracy'] * 40)
        print(f"    {pname:>4} {r['accuracy']:>6.1%} [{bar:<40}] ({r['total']:>6,} frames)")


# ================================================================
# MODEL CONFIGS
# ================================================================
MODEL_CONFIGS = [
    # RVQ baseline
    {
        'name': 'RVQ 4L×128',
        'short': 'rvq_4l',
        'checkpoint': CHECKPOINT_BASE / 'rvq_supervised' / 'best.pt',
        'model_type': 'rvq',
        'num_vq_layers': 4,
        'num_codes': 128,
    },
    # FSQ experiments
    {
        'name': 'FSQ 1K (8×5×5×5)',
        'short': 'fsq_1k',
        'checkpoint': CHECKPOINT_BASE / 'fsq_8x5x5x5_1000' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [8, 5, 5, 5],
    },
    {
        'name': 'FSQ 8K (8×8×5×5×5)',
        'short': 'fsq_8k',
        'checkpoint': CHECKPOINT_BASE / 'fsq_8x8x5x5x5_8000' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [8, 8, 5, 5, 5],
    },
    {
        'name': 'FSQ 1K + BPC',
        'short': 'fsq_1k_bpc',
        'checkpoint': CHECKPOINT_BASE / 'fsq_8x5x5x5_1000_bpc' / 'best.pt',
        'model_type': 'fsq_bpc',
        'fsq_levels': [8, 5, 5, 5],
    },
    {
        'name': 'FSQ 6K 8d (3^8)',
        'short': 'fsq_6k_8d',
        'checkpoint': CHECKPOINT_BASE / 'fsq_3x3x3x3x3x3x3x3_6561' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [3, 3, 3, 3, 3, 3, 3, 3],
    },
    {
        'name': 'FSQ 1K 10d (2^10)',
        'short': 'fsq_1k_10d',
        'checkpoint': CHECKPOINT_BASE / 'fsq_2x2x2x2x2x2x2x2x2x2_1024' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    },
    {
        'name': 'FSQ 43M 16d (3^16)',
        'short': 'fsq_43m_16d',
        'checkpoint': CHECKPOINT_BASE / 'fsq_3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3_43046721' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [3] * 16,
    },
    {
        'name': 'FSQ 1.9Q 32d (3^32)',
        'short': 'fsq_32d',
        'checkpoint': CHECKPOINT_BASE / 'fsq_3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3_1853020188851841' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [3] * 32,
    },
    {
        'name': 'FSQ 3.4e30 64d (3^64)',
        'short': 'fsq_64d',
        'checkpoint': CHECKPOINT_BASE / 'fsq_3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3x3_3433683820292512484657849089281' / 'best.pt',
        'model_type': 'fsq',
        'fsq_levels': [3] * 64,
    },
]


# ================================================================
# SINGLE-CHECKPOINT MODE (--checkpoint)
# ================================================================
def _evaluate_single_checkpoint(args, device):
    """Evaluate one --checkpoint and write a JSON summary to --output_dir."""
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = out_path.parent
    else:
        output_dir = Path(args.output_dir) if args.output_dir else (OUTPUT_DIR / "single_eval")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / 'single_eval_summary.json'

    extract_layers = [int(x) for x in args.extract_layers.split(',')]

    if args.model_type in ('fsq', 'fsq_bpc') and not args.fsq_levels:
        raise ValueError("--fsq_levels required for fsq/fsq_bpc")

    # The CLI values are a fallback. Where the checkpoint carries a fingerprint
    # (or stored training arguments), the model is built from that record and
    # a CLI value that contradicts it is not used; no error is raised.
    # Checkpoints without a fingerprint are built from exactly what is passed
    # here.
    overrides = {
        'fsq_levels': ([int(x) for x in args.fsq_levels.split(',')]
                       if args.fsq_levels else None),
        # The symmetric bound lives in non-persistent buffers, so a checkpoint
        # trained under it loads without error under the asymmetric bound and
        # silently produces different codes. The flag is therefore passed on.
        'preserve_symmetry': args.preserve_symmetry,
        'extract_layers': extract_layers,
        # 3class_cvs / 3class_vus models have a 3-wide BPC head, and
        # load_state_dict(strict=False) does not tolerate a size mismatch.
        'num_bpc_classes': args.num_bpc_classes,
        'num_vq_layers': args.num_vq_layers,
        'num_codes': args.num_codes,
        'hidden_dim': 512,
    }
    model, info = _load_eval_model(
        args.checkpoint, args.model_type, device=device, overrides=overrides,
        bpc_supervision=args.bpc_supervision)
    used = info['config_used']
    logger.info(f"Loaded {args.checkpoint} (step {info.get('step', '?')}; "
                f"fingerprint {'present' if info['had_fingerprint'] else 'absent'})")
    logger.info(f"  built with: "
                f"{ {k: v for k, v in used.items() if v is not None} }")
    fsq_levels = used.get('fsq_levels')

    val_dataset = LibriSpeechPhonemeDataset(args.manifest, max_audio_length=30.0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)
    logger.info(f"Val: {len(val_dataset)} utterances, {len(val_loader)} batches")

    # Streaming rate/entropy instrumentation (scripts/rate_analysis.py).
    if args.model_type in _UNQUANTISED_TYPES:
        # No discrete codes exist (see _UNQUANTISED_TYPES for the per-type
        # reason, which is written into the JSON alongside rate_analysis: null).
        rate_acc = None
    elif args.model_type in ('fsq', 'fsq_bpc'):
        rate_acc = FSQRateAccumulator(fsq_levels)
    elif args.model_type == 'hierarchical':
        # Two VQ layers, each num_codes wide; forward() returns all_indices as
        # [layer1, layer2], which is exactly the RVQ accumulator's input shape.
        rate_acc = RVQRateAccumulator(2, used['num_codes'])
    else:
        rate_acc = RVQRateAccumulator(used['num_vq_layers'], used['num_codes'])

    correct, total, confusion, _ = evaluate_model(
        model, val_loader, device, rate_acc=rate_acc, collect_codes=False)
    overall = sum(correct) / sum(total) * 100 if sum(total) > 0 else 0
    bpc_results = aggregate_to_bpc(correct, total)
    phoneme_results = per_phoneme_table(correct, total)
    logger.info(f"Overall fine accuracy: {overall:.2f}%")
    print_bpc_table(bpc_results, Path(args.checkpoint).parent.name)

    summary = {
        'checkpoint': str(args.checkpoint),
        'manifest': str(args.manifest),
        'model_type': args.model_type,
        'overall_accuracy': overall,
        'seed': args.seed,
        'job_id': args.job_id,
        'eval_job_id': args.eval_job_id,
        'commit_hash': args.commit_hash,
        'bpc': {b: {k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in r.items()} for b, r in bpc_results.items()},
        'phoneme': {p: {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in r.items()} for p, r in phoneme_results.items()},
    }
    # Appended after the accuracy fields, which keep their names, values and
    # positions and do not depend on it.
    if rate_acc is None:
        # Explicit null + reason: distinguishes "this architecture has no codes"
        # from "this JSON predates the rate instrumentation", which a missing
        # key could not.
        summary['rate_analysis'] = None
        summary['rate_analysis_note'] = _UNQUANTISED_TYPES[args.model_type]
        logger.info("=" * 66)
        logger.info("RATE ANALYSIS: null (%s)", summary['rate_analysis_note'])
        logger.info("=" * 66)
    else:
        summary['rate_analysis'] = rate_acc.summary()
        _log_rate_analysis(summary['rate_analysis'])

    # Appended last, like rate_analysis: the configuration the model was
    # built with and whether the checkpoint recorded it. A JSON with
    # had_fingerprint=false comes from a checkpoint whose configuration could
    # not be checked.
    summary['load_provenance'] = {
        'had_fingerprint': info['had_fingerprint'],
        'config_used': {k: v for k, v in used.items()},
        'missing_keys': info['missing_keys'],
        'unexpected_keys': info['unexpected_keys'],
    }

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {out_path}")


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate FSQ/RVQ/BPC checkpoints.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Single-checkpoint mode. If unset, iterates MODEL_CONFIGS.')
    parser.add_argument('--manifest', type=str, default=VAL_MANIFEST)
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['fsq', 'fsq_bpc', 'rvq', 'bpc',
                                 'hierarchical', 'gated', 'fusion', 'identity'])
    parser.add_argument('--fsq_levels', type=str, default=None,
                        help='Comma-separated FSQ levels (e.g. "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3").')
    parser.add_argument('--preserve_symmetry', action='store_true',
                        help='FSQ only. Use the symmetry-preserving bound (Parker '
                             'et al. 2025). It must match how the checkpoint was '
                             'trained: the symmetric geometry lives in '
                             'non-persistent buffers, so a mismatch loads '
                             'cleanly and silently yields the wrong codes. Set '
                             'this for checkpoints from train_fsq_sym.py.')
    parser.add_argument('--extract_layers', type=str, default='4,5,6',
                        help='Transformer layers fused by --model_type '
                             'gated / fusion. Must match training.')
    parser.add_argument('--bpc_supervision', action='store_true',
                        help='--model_type identity only: the checkpoint has a '
                             'second BPC head.')
    parser.add_argument('--num_vq_layers', type=int, default=4)
    parser.add_argument('--num_codes', type=int, default=128)
    parser.add_argument('--num_bpc_classes', type=int, default=8,
                        help='Width of the BPC head for --model_type bpc. Must '
                             'match the checkpoint: 8 for 8class, 3 for '
                             '3class_cvs / 3class_vus.')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None,
                        help='Explicit JSON output path (single-checkpoint mode). '
                             'Overrides the default <output_dir>/single_eval_summary.json.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Provenance only: train-time seed recorded in the JSON.')
    parser.add_argument('--job_id', type=str, default=None,
                        help='Provenance only: job ID of the training run, recorded in the JSON.')
    parser.add_argument('--eval_job_id', type=str, default=None,
                        help='Provenance only: job ID of the evaluation run, recorded in the JSON. '
                             'Distinct from --job_id when eval runs as a separate job from training.')
    parser.add_argument('--commit_hash', type=str, default=None,
                        help='Provenance only: git commit recorded in the JSON.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    if args.checkpoint is not None:
        return _evaluate_single_checkpoint(args, device)

    logger.info("Loading validation data...")
    val_dataset = LibriSpeechPhonemeDataset(VAL_MANIFEST, max_audio_length=30.0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)
    logger.info(f"Val: {len(val_dataset)} utterances, {len(val_loader)} batches")

    all_results = OrderedDict()

    for cfg in MODEL_CONFIGS:
        name = cfg['name']
        ckpt_path = cfg['checkpoint']

        if not ckpt_path.exists():
            logger.warning(f"SKIP {name}: checkpoint not found at {ckpt_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {name}")
        logger.info(f"Checkpoint: {ckpt_path}")

        # Build, verify and load through the one load path. The MODEL_CONFIGS
        # entries below are the fallback configuration for checkpoints without
        # a fingerprint; where a fingerprint exists the model is built from it
        # and a MODEL_CONFIGS value that disagrees with it is not used.
        model, info = _load_eval_model(
            ckpt_path, cfg['model_type'], device=device,
            overrides={'fsq_levels': cfg.get('fsq_levels'),
                       'num_vq_layers': cfg.get('num_vq_layers'),
                       'num_codes': cfg.get('num_codes'),
                       'num_bpc_classes': cfg.get('num_bpc_classes'),
                       'hidden_dim': 512})
        used = info['config_used']
        num_codes = (math.prod(used['fsq_levels'])
                     if cfg['model_type'] in ('fsq', 'fsq_bpc')
                     else used['num_codes'])
        logger.info(f"Loaded checkpoint (step {info.get('step', '?')}; "
                    f"fingerprint {'present' if info['had_fingerprint'] else 'absent'})")

        # Evaluate
        correct, total, confusion, code_data = evaluate_model(model, val_loader, device)

        # Aggregate
        bpc = aggregate_to_bpc(correct, total)
        phoneme = per_phoneme_table(correct, total)
        bpc_cm = aggregate_confusion_bpc(confusion)
        overall = sum(correct) / sum(total) * 100 if sum(total) > 0 else 0

        # Code utilization
        code_util = {}
        code_indices_for_hist = None
        if code_data is not None:
            if code_data.dim() == 1:
                # Flat indices (small codebooks)
                flat = code_data
                unique = flat.unique().numel()
                total_codes = flat.numel()
                code_indices_for_hist = flat

                if num_codes <= 100000:
                    counts = torch.zeros(num_codes)
                    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
                    avg_freq = total_codes / num_codes
                    dead = (counts < 0.01 * avg_freq).sum().item()
                    probs = counts / total_codes
                else:
                    _, inv_counts = flat.unique(return_counts=True)
                    dead = num_codes - unique
                    probs = inv_counts.float() / total_codes

                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                max_entropy = math.log(num_codes)
            else:
                # Multi-dim codes (codebook too large for int64)
                d = code_data.shape[-1]
                total_codes = code_data.shape[0]
                _, _, inv_counts = torch.unique(code_data, dim=0,
                                                return_inverse=True, return_counts=True)
                unique = inv_counts.numel()
                dead = num_codes - unique
                probs = inv_counts.float() / total_codes
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                max_entropy = math.log(num_codes) if num_codes > 1 else 0

            code_util = {
                'num_codes': num_codes,
                'unique_used': unique,
                'utilization_pct': unique / num_codes * 100 if num_codes > 0 else 0,
                'dead_codes': int(dead),
                'dead_pct': dead / num_codes * 100 if num_codes > 0 else 0,
                'entropy': entropy,
                'max_entropy': max_entropy,
                'entropy_ratio': entropy / max_entropy if max_entropy > 0 else 0,
            }

        all_results[name] = {
            'overall_accuracy': overall,
            'bpc': bpc,
            'phoneme': phoneme,
            'bpc_cm': bpc_cm,
            'code_util': code_util,
            'code_indices': code_indices_for_hist,
            'num_codes': num_codes,
        }

        # Print
        logger.info(f"Overall fine accuracy: {overall:.2f}%")
        print_bpc_table(bpc, name)
        if code_util:
            logger.info(
                f"Codebook: {code_util['unique_used']}/{num_codes} used "
                f"({code_util['utilization_pct']:.1f}%), "
                f"Dead: {code_util['dead_codes']} ({code_util['dead_pct']:.1f}%), "
                f"Entropy: {code_util['entropy']:.2f}/{code_util['max_entropy']:.2f} "
                f"({code_util['entropy_ratio']:.1%})"
            )

        # Generate per-model plots
        short = cfg['short']
        plot_error_budget(bpc, f'Error Budget: {name}',
                          OUTPUT_DIR / f'error_budget_{short}.png')
        plot_confusion_heatmap(bpc_cm, BPC_ORDER, f'BPC Confusion: {name}',
                               OUTPUT_DIR / f'confusion_bpc_{short}.png')
        plot_phoneme_confusion(confusion, f'Phoneme Confusion: {name}',
                               OUTPUT_DIR / f'confusion_phoneme_{short}.png')
        if code_indices_for_hist is not None:
            plot_code_histogram(code_indices_for_hist, num_codes, f'Code Usage: {name}',
                                OUTPUT_DIR / f'code_histogram_{short}.png')

        del model
        torch.cuda.empty_cache()

    # ================================================================
    # COMPARISON TABLE
    # ================================================================
    if len(all_results) > 1:
        print("\n" + "=" * 100)
        print("  FSQ vs VQ COMPARISON")
        print("=" * 100)

        # Header
        names = list(all_results.keys())
        header = f"{'Metric':<20}"
        for name in names:
            header += f" {name:>16}"
        print(header)
        print("-" * (20 + 17 * len(names)))

        # Overall accuracy
        row = f"{'Overall Acc':<20}"
        for name in names:
            row += f" {all_results[name]['overall_accuracy']:>15.1f}%"
        print(row)

        # Per-BPC accuracy
        for bpc in BPC_ORDER:
            row = f"{bpc:<20}"
            for name in names:
                if bpc in all_results[name]['bpc']:
                    row += f" {all_results[name]['bpc'][bpc]['accuracy']*100:>15.1f}%"
                else:
                    row += f" {'-':>16}"
            print(row)

        print("-" * (20 + 17 * len(names)))

        # Codebook stats
        row = f"{'Codes (total)':<20}"
        for name in names:
            row += f" {all_results[name]['num_codes']:>16,}"
        print(row)

        row = f"{'Codes (used)':<20}"
        for name in names:
            cu = all_results[name]['code_util']
            if cu:
                row += f" {cu['unique_used']:>16,}"
            else:
                row += f" {'-':>16}"
        print(row)

        row = f"{'Utilization %':<20}"
        for name in names:
            cu = all_results[name]['code_util']
            if cu:
                row += f" {cu['utilization_pct']:>15.1f}%"
            else:
                row += f" {'-':>16}"
        print(row)

        row = f"{'Dead codes %':<20}"
        for name in names:
            cu = all_results[name]['code_util']
            if cu:
                row += f" {cu['dead_pct']:>15.1f}%"
            else:
                row += f" {'-':>16}"
        print(row)

        row = f"{'Entropy ratio':<20}"
        for name in names:
            cu = all_results[name]['code_util']
            if cu:
                row += f" {cu['entropy_ratio']:>15.1%}"
            else:
                row += f" {'-':>16}"
        print(row)

        print("=" * (20 + 17 * len(names)))

    # ================================================================
    # SAVE JSON SUMMARY
    # ================================================================
    summary = {}
    for name in all_results:
        r = all_results[name]
        summary[name] = {
            'overall_accuracy': r['overall_accuracy'],
            'bpc': {b: {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in bpc_data.items()}
                    for b, bpc_data in r['bpc'].items()},
            'phoneme': {p: {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in ph_data.items()}
                        for p, ph_data in r['phoneme'].items()},
            'codebook': r['code_util'],
        }

    json_path = OUTPUT_DIR / 'fsq_comparison_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary JSON: {json_path}")

    # Per-phoneme detail for each model
    print("\n" + "=" * 80)
    print("  PER-PHONEME DETAIL")
    print("=" * 80)
    for name in all_results:
        phoneme_results = all_results[name]['phoneme']
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        for bpc in BPC_ORDER:
            if bpc == 'Silence':
                continue
            members = BPC_MEMBERS.get(bpc, [])
            print(f"\n  [{bpc}] ({len(members)} phonemes)")
            print_per_phoneme(phoneme_results, bpc)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"  Plots: {len(list(OUTPUT_DIR.glob('*.png')))} PNG files")
    print(f"  Data:  fsq_comparison_summary.json")


if __name__ == '__main__':
    main()
