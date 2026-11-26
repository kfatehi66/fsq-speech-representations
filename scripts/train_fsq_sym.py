#!/usr/bin/env python3
"""
Run scripts/train_fsq.py unchanged, but with FSQ(preserve_symmetry=True).

How it works
------------
`scripts/train_fsq.py` has no `--preserve_symmetry` argument. This launcher
rebinds `models.fsq.FSQ` to a subclass that hard-codes
`preserve_symmetry=True`, then executes `scripts.train_fsq` as `__main__`.
train_fsq.py does `from models.fsq import FSQ` at import time, which happens
after the rebind, so it picks up the subclass and nothing else changes.

The two arms of the `[2]^10` comparison are therefore trained by the same code
on the same recipe and, given the same --seed, differ only in the bound. The
command line shows which bound a run used (this script or train_fsq.py), and
the difference from train_fsq.py is exactly one class.

Every argument is forwarded unchanged, so this is a drop-in substitution:

    python scripts/train_fsq_sym.py --fsq_levels "2,2,2,2,2,2,2,2,2,2" \
        --max_steps 30000 --lr 1e-4 --seed 42 --output_dir <dir>

Checkpoints written here load correctly only with preserve_symmetry=True: pass
`--preserve_symmetry` to scripts/evaluate_fsq.py. The state_dict itself is
identical in shape (the symmetric geometry lives in non-persistent buffers), so
loading one under the wrong bound silently produces wrong codes rather than an
error. That is why the flag is given explicitly on both sides, and why the
checkpoints also receive a configuration fingerprint (see
_stamp_fingerprints below).
"""
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models.fsq  # noqa: E402


class SymmetricFSQ(models.fsq.FSQ):
    """models.fsq.FSQ with the symmetry-preserving bound forced on."""

    def __init__(self, input_dim, levels, preserve_symmetry=True):
        # preserve_symmetry is accepted and ignored, so that a caller passing
        # False explicitly still gets the symmetric bound.
        super().__init__(input_dim, levels, preserve_symmetry=True)


def _cli_value(flag, default=None):
    """Read a --flag's value out of sys.argv without re-declaring the parser."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    for a in sys.argv:
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return default


def _stamp_fingerprints():
    """Add the config fingerprint to the checkpoints that train_fsq.py wrote.

    train_fsq.py writes no fingerprint, so it is attached here. This matters
    most for this launcher: preserve_symmetry lives in non-persistent
    buffers, so a symmetric-bound checkpoint loads without complaint under the
    asymmetric bound and silently produces different codes. Without the
    fingerprint there is nothing in the file that records which bound it was
    trained under.
    """
    import torch
    from models.checkpoint_fingerprint import build_fingerprint

    out = _cli_value('--output_dir')
    if not out:
        print('train_fsq_sym.py: no --output_dir, cannot stamp fingerprint',
              flush=True)
        return
    levels_raw = _cli_value('--fsq_levels', '')
    levels = [int(x) for x in levels_raw.replace(',', ' ').split()] or None

    for name in ('best.pt', 'final.pt'):
        path = Path(out) / name
        if not path.exists():
            continue
        ckpt = torch.load(path, map_location='cpu')
        ckpt['config_fingerprint'] = build_fingerprint(
            'ProtoSpeechFSQ', fsq_levels=levels, preserve_symmetry=True,
            hidden_dim=512)
        torch.save(ckpt, path)
        print(f'train_fsq_sym.py: stamped config_fingerprint into {path}',
              flush=True)


def main():
    models.fsq.FSQ = SymmetricFSQ
    print("=" * 60, flush=True)
    print("train_fsq_sym.py: FSQ patched to preserve_symmetry=True", flush=True)
    print("  symmetric bound: Parker et al. 2025, arXiv 2411.19842 §3.2", flush=True)
    print("  running scripts/train_fsq.py unmodified below", flush=True)
    print("=" * 60, flush=True)
    try:
        runpy.run_module('scripts.train_fsq', run_name='__main__', alter_sys=True)
    finally:
        # Runs even if train_fsq.py exits via SystemExit, so a completed run
        # that ends with sys.exit(0) still gets stamped.
        _stamp_fingerprints()


if __name__ == '__main__':
    main()
