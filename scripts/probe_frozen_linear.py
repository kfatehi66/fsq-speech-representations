#!/usr/bin/env python3
"""
Frozen linear phoneme probe over an SSL checkpoint.

The standard readout for self-supervised speech representations: freeze the
encoder, train a single Linear(512, 43) on MFA frame labels, report frame-level
phoneme accuracy. Because the only trainable parameter is the linear map, the
number reflects how linearly separable the phonetic content is in the frozen
representation, not how much a decoder can learn.

What is probed: the context branch, run without masking. That is the
representation through which an SSL encoder is normally evaluated and reused.
The quantised targets are not probed: they exist only to define the pretraining
objective, and probing them would measure the quantiser rather than the encoder
the quantiser was used to train.

The probe is the same for every quantiser branch (same probe, same data, same
schedule). Only the frozen checkpoint differs.

Usage:
    python scripts/probe_frozen_linear.py \
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

from paths import MANIFEST_TRAIN_CLEAN_100, MANIFEST_DEV_CLEAN, OUTPUTS_DIR
from data.phoneme_vocab import VOCAB_SIZE
from models.checkpoint_loading import load_model_from_checkpoint
from scripts.train_fsq import LibriSpeechPhonemeDataset, collate_fn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@torch.no_grad()
def frozen_context(model, wav):
    """Unmasked context features from a frozen encoder."""
    z = model.encode_cnn(wav)
    return model.context(z)          # (B, T, 512), no mask applied


# collate_fn pads labels to the longest label sequence in the batch and the
# waveforms to the longest waveform, so the two maxima can disagree by a frame
# or so on rounding alone. Anything past that is a rate mismatch, not rounding:
# fairseq_padding shifts the frame count by ~1 per second of audio, which over
# the 15 s cap is ~15 frames.
MAX_FRAME_SLACK = 2

_slack_logged = False


def align_frames(logits, labels, *, where, log=None):
    """Reconcile probe frames with label frames, or refuse to guess.

    Taking min(n_logits, n_labels) in both directions is not enough. In one
    direction that is harmless: when the encoder emits more frames than the
    batch has labels, the extra columns are past every utterance's alignment
    and carry no supervision. In the other it is not: truncating labels
    discards real supervised frames, and the accuracy still looks plausible.

    So labels longer than logits are fatal, a small excess of logits is padded
    with the ignore index (which is what those frames already were), and a
    large gap in either direction is fatal because it means the encoder and
    the label pipeline are running at different rates.
    """
    global _slack_logged
    n_logits, n_labels = logits.size(1), labels.size(1)

    if n_labels > n_logits:
        raise RuntimeError(
            f'{where}: the batch carries {n_labels} label frames but the probe '
            f'produced only {n_logits}. Truncating would DISCARD '
            f'{n_labels - n_logits} supervised frame(s) per utterance. This is '
            f'never rounding: the encoder is emitting frames at the wrong rate.')

    if n_logits - n_labels > MAX_FRAME_SLACK:
        raise RuntimeError(
            f'{where}: probe produced {n_logits} frames against {n_labels} '
            f'label frames, a gap of {n_logits - n_labels} > {MAX_FRAME_SLACK}. '
            f'Rounding in collate_fn accounts for at most {MAX_FRAME_SLACK}; a '
            f'gap this size means a frame-rate mismatch. Check the encoder '
            f'frontend: fairseq_padding shifts the count by ~1 frame per '
            f'second of audio.')

    if n_logits > n_labels:
        if not _slack_logged and log is not None:
            _slack_logged = True
            log.info(
                'frame slack: %d logit frame(s) past the last label; padding '
                'with the ignore index. Those frames lie beyond every '
                'utterance\'s alignment, so they carry no supervision either '
                'way. Logged once.', n_logits - n_labels)
        pad = labels.new_full((labels.size(0), n_logits - n_labels), -100)
        labels = torch.cat([labels, pad], dim=1)

    return labels


def evaluate(model, probe, loader, device, max_batches=None):
    model.eval(); probe.eval()
    correct = total = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            wav = batch['waveform'].to(device)
            labels = batch['labels'].to(device)
            logits = probe(frozen_context(model, wav))
            lab = align_frames(logits, labels, where='evaluate', log=logger)
            pred = logits.argmax(-1)
            m = lab >= 0
            correct += int((pred[m] == lab[m]).sum())
            total += int(m.sum())
    return (correct / total * 100) if total else 0.0, total


def main():
    ap = argparse.ArgumentParser(description='Frozen linear phoneme probe')
    ap.add_argument('--checkpoint', type=str, required=True,
                    help='SSL checkpoint from scripts/train_ssl_w2v2.py')
    ap.add_argument('--max_steps', type=int, default=10000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--output_json', type=str, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--job_id', type=str, default=None)
    ap.add_argument('--commit_hash', type=str, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Seed: {args.seed} (cudnn deterministic=True)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # One load path. The encoder is built from the fingerprint that the
    # trainer wrote, not from a reconstruction assembled here out of `args`,
    # which is easy to get wrong: fairseq_padding and no_positional_conv change
    # what the weights mean without changing a single tensor shape, so a
    # strict=False load would silently probe a differently padded encoder.
    model, info = load_model_from_checkpoint(
        args.checkpoint, model_type='ssl', device=device,
        strict_class=True, log=logger)
    ckpt = info['raw']
    quantiser = ckpt.get('quantiser',
                         (ckpt.get('args') or {}).get('quantiser', 'rvq4l'))
    logger.info(f"SSL checkpoint: {args.checkpoint}")
    logger.info(f"  quantiser={quantiser}  step={info.get('step','?')}  "
                f"val_loss={ckpt.get('val_loss', ckpt.get('best_val_loss','?'))}  "
                f"fingerprint={'present' if info['had_fingerprint'] else 'absent'}")

    model = model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # The only trainable parameters.
    probe = nn.Linear(model.hidden_dim, VOCAB_SIZE).to(device)
    logger.info(f"Probe: Linear({model.hidden_dim}, {VOCAB_SIZE}), "
                f"{sum(p.numel() for p in probe.parameters()):,} trainable params")

    train_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_TRAIN_CLEAN_100), 15.0, 1.0),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_DEV_CLEAN), 15.0, 1.0),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    optimizer = AdamW(probe.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.max_steps, pct_start=0.08)

    step, best = 0, 0.0
    for epoch in range(10000):
        for batch in train_loader:
            if step >= args.max_steps:
                break
            wav = batch['waveform'].to(device)
            labels = batch['labels'].to(device)

            feats = frozen_context(model, wav)      # no grad through the encoder
            logits = probe(feats)
            labels = align_frames(logits, labels, where='probe training',
                                  log=logger)
            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                labels.reshape(-1), ignore_index=-100)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 500 == 0:
                logger.info(f"Step {step} | CE {loss.item():.4f}")
            if step % 2000 == 0 and step > 0:
                acc, n = evaluate(model, probe, val_loader, device, max_batches=100)
                logger.info(f"[VAL] Step {step} | dev-clean probe acc {acc:.2f}% "
                            f"({n:,} frames)")
                best = max(best, acc)
                probe.train()
            step += 1
        if step >= args.max_steps:
            break

    acc, n = evaluate(model, probe, val_loader, device)
    logger.info("=" * 60)
    logger.info(f"FROZEN LINEAR PROBE ({quantiser})")
    logger.info(f"  dev-clean frame accuracy: {acc:.2f}%  ({n:,} frames)")
    logger.info("=" * 60)

    summary = {
        'ssl_checkpoint': str(args.checkpoint),
        'quantiser': quantiser,
        'ssl_step': info.get('step'),
        'had_fingerprint': info['had_fingerprint'],
        'probe_steps': args.max_steps,
        'probe_lr': args.lr,
        'dev_clean_probe_accuracy': acc,
        'dev_clean_frames': n,
        'best_intermediate_accuracy': max(best, acc),
        'seed': args.seed,
        'job_id': args.job_id,
        'commit_hash': args.commit_hash,
    }
    out = Path(args.output_json) if args.output_json else (
        Path(OUTPUTS_DIR) / 'ssl_probe' / f'probe_{quantiser}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved: {out}")


if __name__ == '__main__':
    main()
