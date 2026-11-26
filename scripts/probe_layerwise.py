#!/usr/bin/env python3
"""
Layer-wise frozen probes over an SSL checkpoint.

`probe_frozen_linear.py` probes one representation, the final context output.
That is the standard readout, but it cannot distinguish "this encoder learned
nothing useful" from "this encoder learned something useful and then discarded
it in the last layers", which lead to opposite conclusions. This script
therefore probes every tap:

    conv output z  ->  transformer layer 1 ... 6

with two heads per tap:

    Linear                 : raw linear separability
    LayerNorm + Linear     : separability after the scale/shift a downstream
                             consumer would apply anyway. A representation with
                             a large uninformative mean or wildly varying
                             per-dimension scale can look unprobeable bare and
                             be fine normalised, and that difference is itself
                             diagnostic.

14 probes per checkpoint, all trained on the same frozen forward pass, so the
whole sweep costs about one ordinary probe run rather than fourteen.

Also emits the per-tap feature statistics (per-dimension mean/std, mean L2
norm, count of near-dead dimensions).

Usage:
    python scripts/probe_layerwise.py \
        --checkpoint <ssl_dir>/best.pt --max_steps 10000 --seed 42 \
        --output_json <path>
"""
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from paths import MANIFEST_TRAIN_CLEAN_100, MANIFEST_DEV_CLEAN
from data.phoneme_vocab import VOCAB_SIZE
from models.checkpoint_loading import load_model_from_checkpoint
from scripts.probe_frozen_linear import align_frames
from scripts.train_fsq import LibriSpeechPhonemeDataset, collate_fn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def taps(model, wav):
    """Every intermediate representation, unmasked, in forward order.

    Mirrors SSLEncoder.context() exactly (positional conv, pre-norm, then each
    Transformer layer), so 'layer_6' here is the same tensor that context()
    returns before its final layer_norm.
    """
    out = {}
    z = model.encode_cnn(wav)
    out['conv'] = z

    h = z
    if model.pos_conv is not None:
        h = model.pos_conv(h)
    h = model.pre_transformer_norm(h)
    for i, layer in enumerate(model.transformer_layers, start=1):
        h = layer(h)
        out[f'layer_{i}'] = h
    return out


class ProbeHead(nn.Module):
    """Linear, or LayerNorm then Linear."""

    def __init__(self, dim, n_classes, normed):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if normed else None
        self.fc = nn.Linear(dim, n_classes)

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        return self.fc(x)


def feature_stats(h):
    """Per-dimension mean/std, mean L2 norm, near-dead dimension count."""
    flat = h.reshape(-1, h.shape[-1]).float()
    mu, sd = flat.mean(0), flat.std(0)
    return {
        'per_dim_mean_avg': float(mu.mean()),
        'per_dim_mean_min': float(mu.min()),
        'per_dim_mean_max': float(mu.max()),
        'per_dim_std_avg': float(sd.mean()),
        'per_dim_std_min': float(sd.min()),
        'per_dim_std_max': float(sd.max()),
        'mean_l2_norm_per_frame': float(flat.norm(dim=-1).mean()),
        'n_dims_std_below_1e-3': int((sd < 1e-3).sum()),
        'n_dims': int(flat.shape[-1]),
    }


@torch.no_grad()
def evaluate(model, heads, loader, device, max_batches=None):
    for h in heads.values():
        h.eval()
    correct = {k: 0 for k in heads}
    total = 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        wav = batch['waveform'].to(device)
        labels = batch['labels'].to(device)
        reps = taps(model, wav)
        any_logits = next(iter(heads.values()))(reps['conv'])
        lab = align_frames(any_logits, labels, where='layerwise eval', log=logger)
        mask = lab >= 0
        total += int(mask.sum())
        for name, head in heads.items():
            tap = name.rsplit('|', 1)[0]
            pred = head(reps[tap]).argmax(-1)
            correct[name] += int((pred[mask] == lab[mask]).sum())
    for h in heads.values():
        h.train()
    return {k: (correct[k] / total * 100 if total else 0.0) for k in heads}, total


def main():
    ap = argparse.ArgumentParser(description='Layer-wise frozen probes')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--max_steps', type=int, default=10000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--eval_batches', type=int, default=None,
                    help='Cap dev-clean batches during intermediate evals.')
    ap.add_argument('--output_json', default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--job_id', default=None)
    ap.add_argument('--commit_hash', default=None)
    ap.add_argument('--label', default=None, help='Name for this arm, written to the output JSON as "label".')
    ap.add_argument('--taps', default=None,
                    help='Comma-separated subset of taps to probe, e.g. '
                         '"layer_1,layer_6". Default: every tap. Restricting '
                         'the set is what makes a 20-checkpoint trajectory '
                         'sweep affordable: the frozen forward pass is shared '
                         'either way, but each head costs an optimiser slot '
                         'and a term in the summed loss.')
    ap.add_argument('--heads', default='linear,lnlinear',
                    help="Which head kinds to fit: 'linear', 'lnlinear', or both.")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Seed: {args.seed} (cudnn deterministic=True)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # One load path, built from the fingerprint rather than from a
    # reconstruction assembled here out of `args`.
    model, info = load_model_from_checkpoint(
        args.checkpoint, model_type='ssl', device=device,
        strict_class=True, log=logger)
    ckpt = info['raw']
    quantiser = ckpt.get('quantiser',
                         (ckpt.get('args') or {}).get('quantiser', 'rvq4l'))
    logger.info(f"SSL checkpoint: {args.checkpoint}")
    logger.info(f"  quantiser={quantiser}  step={info.get('step','?')}  "
                f"untrained={ckpt.get('untrained', False)}  "
                f"fingerprint={'present' if info['had_fingerprint'] else 'absent'}")

    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False

    all_taps = ['conv'] + [f'layer_{i}'
                           for i in range(1, len(model.transformer_layers) + 1)]
    tap_names = ([t.strip() for t in args.taps.split(',') if t.strip()]
                 if args.taps else list(all_taps))
    unknown = [t for t in tap_names if t not in all_taps]
    if unknown:
        raise ValueError(f'--taps: {unknown} not among {all_taps}')
    kinds = [k.strip() for k in args.heads.split(',') if k.strip()]
    if any(k not in ('linear', 'lnlinear') for k in kinds):
        raise ValueError("--heads takes 'linear' and/or 'lnlinear'")

    heads = nn.ModuleDict()
    for tap in tap_names:
        for kind, normed in (('linear', False), ('lnlinear', True)):
            if kind in kinds:
                heads[f'{tap}|{kind}'] = ProbeHead(model.hidden_dim, VOCAB_SIZE,
                                                   normed)
    heads = heads.to(device)
    n_params = sum(p.numel() for p in heads.parameters())
    logger.info(f"{len(heads)} probe heads over {len(tap_names)} taps "
                f"({n_params:,} trainable params); the encoder is frozen")

    train_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_TRAIN_CLEAN_100), 15.0, 1.0),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_DEV_CLEAN), 15.0, 1.0),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    optimizer = AdamW(heads.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.max_steps, pct_start=0.08)

    stats, step = None, 0
    for epoch in range(10000):
        for batch in train_loader:
            if step >= args.max_steps:
                break
            wav = batch['waveform'].to(device)
            labels = batch['labels'].to(device)
            reps = taps(model, wav)

            if stats is None:                       # feature stats, first batch only
                stats = {t: feature_stats(reps[t]) for t in tap_names}

            # Any head will do for the frame-count reconciliation: every tap
            # runs at the same rate, and align_frames only reads the time
            # dimension. Take the first rather than assume conv|linear exists,
            # which it does not when --taps restricts the sweep.
            probe_name = next(iter(heads))
            probe_logits = heads[probe_name](reps[probe_name.rsplit('|', 1)[0]])
            lab = align_frames(probe_logits, labels, where='layerwise training',
                               log=logger)

            loss = 0.0
            for name, head in heads.items():
                tap = name.rsplit('|', 1)[0]
                loss = loss + F.cross_entropy(
                    head(reps[tap]).reshape(-1, VOCAB_SIZE),
                    lab.reshape(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 1000 == 0:
                logger.info(f"Step {step} | summed CE {float(loss):.4f}")
            step += 1
        if step >= args.max_steps:
            break

    logger.info("Final dev-clean evaluation over every tap ...")
    acc, n_frames = evaluate(model, heads, val_loader, device,
                             max_batches=args.eval_batches)

    logger.info("")
    logger.info(f"{'tap':<10}{'Linear':>12}{'LN+Linear':>12}{'delta':>9}")
    logger.info('-' * 43)
    for tap in tap_names:
        a, b = acc.get(f'{tap}|linear'), acc.get(f'{tap}|lnlinear')
        cells = ''.join(f'{v:>11.2f}%' if v is not None else f"{'-':>12}"
                        for v in (a, b))
        delta = f'{b - a:>+9.2f}' if (a is not None and b is not None) else f"{'-':>9}"
        logger.info(f"{tap:<10}{cells}{delta}")

    best = max(acc.items(), key=lambda kv: kv[1])
    logger.info("")
    logger.info(f"best probe: {best[0]} = {best[1]:.2f}%  ({n_frames:,} frames)")

    result = {
        'label': args.label,
        'ssl_checkpoint': args.checkpoint,
        'quantiser': quantiser,
        'ssl_step': info.get('step'),
        'had_fingerprint': info['had_fingerprint'],
        'untrained': bool(ckpt.get('untrained', False)),
        'probe_steps': args.max_steps,
        'probe_lr': args.lr,
        'taps': tap_names,
        'heads': kinds,
        'accuracy': acc,
        'best_tap': best[0],
        'best_accuracy': best[1],
        # The last Transformer layer, not the last selected tap: with --taps
        # the two can differ, and the SSL selection rule is read off this
        # field.
        'final_layer_linear': acc.get(f'{all_taps[-1]}|linear'),
        'dev_clean_frames': n_frames,
        'feature_stats': stats,
        'seed': args.seed,
        'job_id': args.job_id,
        'commit_hash': args.commit_hash,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, 'w') as fh:
            json.dump(result, fh, indent=2)
        logger.info(f"wrote {args.output_json}")


if __name__ == '__main__':
    main()
