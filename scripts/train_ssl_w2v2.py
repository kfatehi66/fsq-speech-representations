#!/usr/bin/env python3
r"""
wav2vec 2.0-style contrastive pretraining with a swappable quantiser.

The quantiser produces the prediction targets of a self-supervised objective
instead of the bottleneck of a supervised classifier, so FSQ and RVQ can be
compared in that role.

    CNN(z) --> quantise --> targets q
           \--> span-mask --> +positional conv --> Transformer --> context c
    L = L_m(c, q)  [+ alpha * L_d, learned-codebook branches only]

Everything structural lives in models/ssl_encoder.py and
losses/w2v2_contrastive.py; this file is the loop.

Data pipeline: imported from scripts/train_fsq.py, so the SSL runs see the
same dataset as the supervised runs (same manifest, same TextGrid filter, same
15 s crop, same collate). Frame labels are loaded and ignored here;
scripts/probe_frozen_linear.py uses them.

Diversity loss: applied to rvq4l and gumbel_pq only. The fsq16d branch has no
learned codebook, so there is nothing for L_d to regularise. See
losses/w2v2_contrastive.py.

Usage:
    python scripts/train_ssl_w2v2.py --quantiser rvq4l --max_steps 20000 \
        --seed 42 --output_dir <dir>
"""
import argparse
import json
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from paths import CHECKPOINTS_DIR, MANIFEST_TRAIN_CLEAN_100, MANIFEST_DEV_CLEAN
from models.checkpoint_fingerprint import build_fingerprint
from models.ssl_encoder import SSLEncoder, QUANTISER_CHOICES
from losses.w2v2_contrastive import W2V2ContrastiveLoss, CodebookDiversityLoss

# Identical data pipeline to the supervised runs (see module docstring).
from scripts.train_fsq import LibriSpeechPhonemeDataset, collate_fn

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


def quantiser_stats(q_out, quantiser_name):
    """Per-branch codebook health.

    RVQ / gumbel: per-group perplexity and used-code count.
    FSQ:          per-dimension level usage, the FSQ analogue of "dead codes".
    """
    stats = {}
    if quantiser_name == 'fsq16d':
        codes = q_out.get('codes')
        if codes is not None:
            flat = codes.reshape(-1, codes.shape[-1])
            stats['fsq_levels_used_per_dim'] = [
                int(flat[:, i].unique().numel()) for i in range(flat.shape[1])]
            stats['fsq_unique_codes'] = int(
                torch.unique(flat, dim=0).shape[0])
    else:
        perp, used = [], []
        for idx in q_out['indices']:
            f = idx.reshape(-1)
            counts = torch.bincount(f, minlength=q_out['num_entries']).float()
            p = counts / counts.sum()
            nz = p[p > 0]
            perp.append(float(torch.exp(-(nz * nz.log()).sum())))
            used.append(int((counts > 0).sum()))
        stats['perplexity_per_group'] = perp
        stats['used_codes_per_group'] = used
    return stats


def main():
    ap = argparse.ArgumentParser(description='wav2vec2-style SSL pretraining')
    ap.add_argument('--quantiser', type=str, default='rvq4l',
                    choices=list(QUANTISER_CHOICES))
    ap.add_argument('--max_steps', type=int, default=100000)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_negatives', type=int, default=100,
                    help='K distractors per masked frame (wav2vec 2.0: 100).')
    ap.add_argument('--temperature', type=float, default=0.1,
                    help='kappa in the InfoNCE (wav2vec 2.0: 0.1).')
    ap.add_argument('--mask_prob', type=float, default=0.065)
    ap.add_argument('--mask_length', type=int, default=10)
    ap.add_argument('--diversity_weight', type=float, default=0.1,
                    help='alpha on L_d. Ignored for fsq16d, which has no '
                         'learned codebook (see the module docstring).')
    ap.add_argument('--no_positional_conv', action='store_true',
                    help='For debugging only. Removes the relative positional conv, '
                         'which makes the encoder permutation-equivariant and '
                         'the objective degenerate. See models/ssl_encoder.py.')
    ap.add_argument('--fairseq_padding', action='store_true',
                    help="For ablations only. Drops the padding on the kernel-3 conv "
                         "layers, reproducing wav2vec 2.0's frontend (49 frames "
                         'per second of audio instead of 50). The default keeps '
                         'the frontend byte-identical to the supervised '
                         'ProtoSpeech models, which is what makes the SSL and '
                         'supervised numbers comparable.')
    ap.add_argument('--fsq_levels', type=str, default='3,' * 15 + '3')
    ap.add_argument('--num_vq_layers', type=int, default=4)
    ap.add_argument('--num_codes', type=int, default=128)
    ap.add_argument('--checkpoint_every', type=int, default=0,
                    help='Also write step_<N>.pt every N steps (0 = off). The '
                         '100k full runs need this: without intermediate '
                         'checkpoints there is no way to probe how the '
                         'representation evolves, and re-running to find out '
                         'costs another 100k steps.')
    ap.add_argument('--log_every', type=int, default=100)
    ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--output_dir', type=str, default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    # Seed block: identical to the one in scripts/train_fsq.py.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Seed: {args.seed} (cudnn deterministic=True, benchmark=False)")
    else:
        logger.info("Seed: unseeded (OS entropy)")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    output_dir = Path(args.output_dir) if args.output_dir else (
        CHECKPOINTS_DIR / f"ssl_w2v2_{args.quantiser}")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("wav2vec2-STYLE SSL PRETRAINING")
    logger.info("=" * 60)
    logger.info(f"Quantiser: {args.quantiser} (on the CNN output z)")
    logger.info(f"Mask: prob={args.mask_prob}, length={args.mask_length}")
    logger.info(f"Negatives: K={args.num_negatives} (within-utterance, masked "
                f"positions only), kappa={args.temperature}")
    logger.info(f"Steps: {args.max_steps}  lr: {args.lr}  batch: {args.batch_size}")
    logger.info(f"Output dir: {output_dir}")

    quant_kw = {}
    if args.quantiser == 'fsq16d':
        quant_kw['fsq_levels'] = [int(x) for x in args.fsq_levels.split(',')]
    elif args.quantiser == 'rvq4l':
        quant_kw['num_vq_layers'] = args.num_vq_layers
        quant_kw['num_codes'] = args.num_codes

    model = SSLEncoder(quantiser=args.quantiser,
                       use_positional_conv=not args.no_positional_conv,
                       fairseq_padding=args.fairseq_padding,
                       **quant_kw).to(device)

    # Config fingerprint: fairseq_padding and no_positional_conv change what the weights
    # mean without changing any parameter shape, so the probe has to be able
    # to verify it rebuilt the same encoder.
    FINGERPRINT = build_fingerprint(model, args, hidden_dim=model.hidden_dim)

    contrastive = W2V2ContrastiveLoss(num_negatives=args.num_negatives,
                                      temperature=args.temperature)

    # L_d is constructed only where there is a learned codebook to regularise.
    use_diversity = args.quantiser in ('rvq4l', 'gumbel_pq')
    if use_diversity:
        q = model.quantiser
        diversity = CodebookDiversityLoss(num_groups=q.num_groups if hasattr(q, 'num_groups')
                                          else args.num_vq_layers,
                                          num_entries=q.num_entries if hasattr(q, 'num_entries')
                                          else args.num_codes).to(device)
        logger.info(f"Diversity loss L_d: on, alpha={args.diversity_weight}")
    else:
        diversity = None
        logger.info("Diversity loss L_d: off. FSQ has no learned codebook to "
                    "collapse, so there is no distribution for L_d to act on. "
                    "This asymmetry is intended.")

    train_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_TRAIN_CLEAN_100), 15.0, 1.0),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(
        LibriSpeechPhonemeDataset(str(MANIFEST_DEV_CLEAN), 15.0, 1.0),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr,
                           total_steps=args.max_steps, pct_start=0.08)

    # Masking is drawn from a dedicated generator so it is reproducible under
    # --seed independently of dataloader shuffling order.
    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device).manual_seed(args.seed + 1)

    history = []
    global_step = 0
    best_val = float('inf')

    logger.info("Starting SSL pretraining...")
    for epoch in range(10000):
        model.train()
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            wav = batch['waveform'].to(device)

            if args.quantiser == 'gumbel_pq':
                model.quantiser.set_num_updates(global_step)

            out = model(wav, mask_prob=args.mask_prob,
                        mask_length=args.mask_length, generator=gen)

            l_m, cstats = contrastive(out['context'], out['target'],
                                      out['mask'], generator=gen)
            total = l_m
            dstats = {}
            if diversity is not None:
                probs = out['quantiser_out']['diversity_probs']
                l_d, dstats = diversity(probs)
                total = total + args.diversity_weight * l_d

            if torch.isnan(total):
                logger.warning(f"Step {global_step}: NaN loss, skipping")
                continue

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % args.log_every == 0:
                with torch.no_grad():
                    qs = quantiser_stats(out['quantiser_out'], args.quantiser)
                msg = (f"Step {global_step} | L_m {cstats['contrastive_loss']:.4f} "
                       f"| acc {cstats['accuracy']:.1%} "
                       f"| masked {cstats['n_masked']}")
                if dstats:
                    msg += (f" | L_d {dstats['diversity_loss']:.4f} "
                            f"| perp {['%.1f' % p for p in dstats['perplexity_per_group']]}")
                if 'fsq_levels_used_per_dim' in qs:
                    msg += (f" | levels/dim {qs['fsq_levels_used_per_dim']} "
                            f"| uniq {qs['fsq_unique_codes']}")
                elif 'perplexity_per_group' in qs:
                    msg += f" | used {qs['used_codes_per_group']}"
                logger.info(msg)

                history.append({
                    'step': global_step,
                    'contrastive_loss': cstats['contrastive_loss'],
                    'contrastive_accuracy': cstats['accuracy'],
                    'duplicate_negative_rate': cstats.get('duplicate_negative_rate'),
                    'diversity_loss': dstats.get('diversity_loss'),
                    'quantiser_stats': qs,
                    'lr': scheduler.get_last_lr()[0],
                })

            if global_step % args.val_every == 0 and global_step > 0:
                model.eval()
                vl, vacc, nb = 0.0, 0.0, 0
                with torch.no_grad():
                    for i, vb in enumerate(val_loader):
                        if i >= 50:      # bounded: this is a monitor, not a metric
                            break
                        vout = model(vb['waveform'].to(device),
                                     mask_prob=args.mask_prob,
                                     mask_length=args.mask_length)
                        vloss, vs = contrastive(vout['context'], vout['target'],
                                                vout['mask'])
                        vl += vs['contrastive_loss']; vacc += vs['accuracy']; nb += 1
                vl, vacc = vl / max(nb, 1), vacc / max(nb, 1)
                logger.info(f"[VAL] Step {global_step} | L_m {vl:.4f} | acc {vacc:.1%}")
                history.append({'step': global_step, 'val_contrastive_loss': vl,
                                'val_contrastive_accuracy': vacc})

                if vl < best_val:
                    best_val = vl
                    torch.save({'model_state_dict': model.state_dict(),
                                'config_fingerprint': FINGERPRINT,
                                'step': global_step, 'val_loss': vl,
                                'quantiser': args.quantiser,
                                'args': vars(args)}, output_dir / 'best.pt')
                    logger.info(f"Saved best (val L_m={vl:.4f})")
                model.train()

            if (args.checkpoint_every and global_step > 0
                    and global_step % args.checkpoint_every == 0):
                # Codebook statistics at the checkpoint, computed afresh rather
                # than taken from the `qs` of the last log_every step. A slow
                # collapse, the failure mode expected at alpha=0, cannot show
                # within a 20k-step run, so each checkpoint carries its own
                # per-layer perplexity and the trajectory can be read
                # afterwards.
                with torch.no_grad():
                    ckpt_qs = quantiser_stats(out['quantiser_out'], args.quantiser)
                torch.save({'model_state_dict': model.state_dict(),
                            'config_fingerprint': FINGERPRINT,
                            'step': global_step, 'quantiser': args.quantiser,
                            'quantiser_stats': ckpt_qs,
                            'diversity_weight': args.diversity_weight,
                            'args': vars(args)},
                           output_dir / f'step_{global_step}.pt')
                if 'perplexity_per_group' in ckpt_qs:
                    logger.info(
                        "CHECKPOINT step_%d.pt | per-layer perplexity %s | "
                        "used codes %s (of %d)", global_step,
                        ['%.1f' % x for x in ckpt_qs['perplexity_per_group']],
                        ckpt_qs['used_codes_per_group'],
                        out['quantiser_out']['num_entries'])
                elif 'fsq_levels_used_per_dim' in ckpt_qs:
                    logger.info(
                        "CHECKPOINT step_%d.pt | levels/dim %s | unique codes %d",
                        global_step, ckpt_qs['fsq_levels_used_per_dim'],
                        ckpt_qs['fsq_unique_codes'])
                else:
                    logger.info(f"Wrote intermediate checkpoint step_{global_step}.pt")

            global_step += 1

        if global_step >= args.max_steps:
            break

    torch.save({'model_state_dict': model.state_dict(), 'step': global_step,
                'config_fingerprint': FINGERPRINT,
                'best_val_loss': best_val, 'quantiser': args.quantiser,
                'args': vars(args)}, output_dir / 'final.pt')
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"SSL pretraining complete: {global_step} steps, "
                f"best val L_m {best_val:.4f}")
    logger.info(f"History: {output_dir / 'history.json'}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
