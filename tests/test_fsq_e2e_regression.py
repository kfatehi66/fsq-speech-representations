#!/usr/bin/env python3
"""
End-to-end regression: the even-L correction of models/fsq.py leaves the
ternary results unchanged.

Loads a trained FSQ-16d checkpoint and evaluates the same dev-clean utterances
twice, once through models/fsq.py and once through models/fsq_legacy.py (the
implementation without the even-L offset), with identical weights. Asserts
bit-identical logits, predictions, codes and accuracy.

A failure means that the correction has changed the odd-L path, and the
multi-seed results of the paper would no longer be reproducible.

Usage:
    python tests/test_fsq_e2e_regression.py [--n_utts 50] [--checkpoint PATH]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paths import ARTEFACTS_ROOT, MANIFEST_DEV_CLEAN          # noqa: E402
from models.fsq_legacy import FSQ as LegacyFSQ                 # noqa: E402
from scripts.evaluate_fsq import (                             # noqa: E402
    ProtoSpeechFSQ, LibriSpeechPhonemeDataset, collate_fn, VOCAB_SIZE,
)

DEFAULT_CKPT = (Path(ARTEFACTS_ROOT) / 'checkpoints'
                / 'fsq_16d_seed42' / 'best.pt')
FSQ_LEVELS = [3] * 16   # published FSQ-16d config (all odd -> must not move)


def _swap_in_legacy_fsq(model):
    """Return a deep copy of `model` with models.fsq.FSQ -> fsq_legacy.FSQ."""
    legacy_model = copy.deepcopy(model)
    old = LegacyFSQ(input_dim=legacy_model.fsq.input_dim,
                    levels=list(legacy_model.fsq.levels))
    old.project_down.load_state_dict(legacy_model.fsq.project_down.state_dict())
    old.project_up.load_state_dict(legacy_model.fsq.project_up.state_dict())
    legacy_model.fsq = old
    return legacy_model.eval()


def _encode(model, loader, device):
    """Run the (untouched) CNN + Transformer once; yield hidden states + labels.

    The encoder is byte-identical between the two variants, so running it twice
    would only burn compute. Comparing the quantisers on identical real hidden
    states is the strictly stronger test: it isolates the quantiser.
    """
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for batch in loader:
            wav = batch['waveform'].to(device)
            if wav.dim() == 2:
                wav = wav.unsqueeze(1)
            hidden = model.feature_encoder(wav).transpose(1, 2)
            for layer in model.transformer_layers:
                hidden = layer(hidden)
            hidden = model.layer_norm(hidden)
            out.append((hidden.cpu(), batch['labels'].cpu()))
    return out


def _quantise_and_score(model, encoded, device):
    """Push cached hidden states through model.fsq + fine_head."""
    model.eval().to(device)
    correct = np.zeros(VOCAB_SIZE, dtype=np.int64)
    total = np.zeros(VOCAB_SIZE, dtype=np.int64)
    logits_all, codes_all = [], []

    with torch.no_grad():
        for hidden, labels in encoded:
            fsq_out = model.fsq(hidden.to(device))
            logits = model.fine_head(fsq_out['quantized'])

            logits_all.append(logits.cpu())
            codes_all.append(fsq_out['codes'].cpu())

            preds = logits.argmax(dim=-1)
            n = min(preds.size(1), labels.size(1))
            pf = preds[:, :n].reshape(-1).cpu().numpy()
            lf = labels[:, :n].reshape(-1).numpy()
            valid = lf >= 0
            vl, vp = lf[valid], pf[valid]
            for c in range(VOCAB_SIZE):
                m = vl == c
                total[c] += m.sum()
                correct[c] += (vp[m] == c).sum()

    acc = correct.sum() / total.sum() * 100 if total.sum() else 0.0
    return acc, logits_all, codes_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default=str(DEFAULT_CKPT))
    ap.add_argument('--manifest', type=str, default=str(MANIFEST_DEV_CLEAN))
    ap.add_argument('--n_utts', type=int, default=50)
    # Defaults are sized for a CPU-only machine: the first conv layer expands to
    # 512 channels at 1/5 the sample rate, so batch 8 x 30 s needs >64 GB of
    # memory. Raise both when running on a GPU.
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--max_audio_length', type=float, default=10.0)
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"SKIP: checkpoint not found: {ckpt_path}")
        return 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    print(f"checkpoint : {ckpt_path}")
    print(f"manifest   : {args.manifest}")
    print(f"device     : {device}   n_utts: {args.n_utts}")

    model_new = ProtoSpeechFSQ(fsq_levels=FSQ_LEVELS)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    missing, unexpected = model_new.load_state_dict(
        ckpt['model_state_dict'], strict=False)
    print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    # Only FSQ geometry buffers may be missing/unexpected; no learned weights.
    for k in list(missing) + list(unexpected):
        assert 'fsq._' in k, f"non-buffer key mismatch on load: {k}"

    model_old = _swap_in_legacy_fsq(model_new)

    ds = LibriSpeechPhonemeDataset(args.manifest,
                                   max_audio_length=args.max_audio_length)
    ds = Subset(ds, range(min(args.n_utts, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_fn)

    print("encoding (shared CNN + Transformer, run once)...", flush=True)
    encoded = _encode(model_new, loader, device)
    print(f"  {len(encoded)} batches encoded", flush=True)

    acc_new, log_new, cod_new = _quantise_and_score(model_new, encoded, device)
    acc_old, log_old, cod_old = _quantise_and_score(model_old, encoded, device)

    print(f"\naccuracy (models/fsq.py)        : {acc_new:.10f}%")
    print(f"accuracy (models/fsq_legacy.py) : {acc_old:.10f}%")

    ok = True
    if acc_new != acc_old:
        ok = False
        print(f"FAIL: accuracy differs by {abs(acc_new - acc_old):.3e} pp")

    for i, (a, b) in enumerate(zip(cod_new, cod_old)):
        if not torch.equal(a, b):
            ok = False
            n_diff = int((a != b).sum())
            print(f"FAIL: codes differ in batch {i} ({n_diff} elements)")
            break

    for i, (a, b) in enumerate(zip(log_new, log_old)):
        if not torch.equal(a, b):
            ok = False
            print(f"FAIL: logits differ in batch {i} "
                  f"(max |delta| = {(a - b).abs().max():.3e})")
            break

    n_frames = sum(c.shape[0] * c.shape[1] for c in cod_new)
    if ok:
        print(f"\nPASS: bit-identical over {n_frames:,} frames "
              f"({len(cod_new)} batches): accuracy, logits and codes all equal.")
        return 0
    print("\nFAIL: the even-L correction changed odd-L behaviour, so odd-L results would not reproduce.")
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
