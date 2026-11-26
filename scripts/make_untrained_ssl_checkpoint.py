#!/usr/bin/env python3
"""
Write a randomly initialised, seeded SSL encoder as a checkpoint.

A frozen probe on this encoder measures what the architecture alone provides,
before any pre-training: the untrained-encoder floor of the self-supervised
probes. The encoder is seeded in the same way as scripts/train_ssl_w2v2.py
seeds itself.

The quantiser is the RVQ branch (4 layers of 128 codes), the default quantiser
of scripts/train_ssl_w2v2.py. The probes read the CNN output and the context
branch, which do not depend on the quantiser, so the floor is a property of
the encoder rather than of the quantiser choice.

Usage:
    python scripts/make_untrained_ssl_checkpoint.py --output_dir <dir> [--seed 42]

The output is <dir>/best.pt, which both probe scripts read with --checkpoint.
"""
import argparse
import os
import random

import numpy as np
import torch

from models.checkpoint_fingerprint import build_fingerprint
from models.ssl_encoder import SSLEncoder
from paths import CHECKPOINTS_DIR


def main():
    ap = argparse.ArgumentParser(
        description='Write a randomly initialised, seeded SSL encoder as best.pt.')
    ap.add_argument('--output_dir', type=str, default=None,
                    help='Directory for best.pt, created if needed. Default: '
                         'CHECKPOINTS_DIR/ssl_untrained_floor_seed<seed>.')
    ap.add_argument('--seed', type=int, default=42,
                    help='Seed of the initialisation (default: 42).')
    cli = ap.parse_args()

    out_dir = cli.output_dir or str(CHECKPOINTS_DIR / f'ssl_untrained_floor_seed{cli.seed}')
    os.makedirs(out_dir, exist_ok=True)

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = SSLEncoder(quantiser='rvq4l', num_vq_layers=4, num_codes=128)

    args = {'quantiser': 'rvq4l', 'num_vq_layers': 4, 'num_codes': 128,
            'no_positional_conv': False, 'fairseq_padding': False, 'seed': cli.seed,
            'max_steps': 0}
    torch.save({
        'model_state_dict': model.state_dict(),
        'config_fingerprint': build_fingerprint(model, args,
                                                hidden_dim=model.hidden_dim),
        'step': 0,
        'quantiser': 'rvq4l',
        'untrained': True,
        'args': args,
    }, f'{out_dir}/best.pt')
    n = sum(p.numel() for p in model.parameters())
    print(f'wrote {out_dir}/best.pt  ({n/1e6:.2f}M parameters, 0 training steps)',
          flush=True)


if __name__ == '__main__':
    main()
