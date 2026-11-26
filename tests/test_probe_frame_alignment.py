#!/usr/bin/env python3
"""
Tests for probe_frozen_linear.align_frames.

The probe compares encoder frames against MFA label frames. `collate_fn` pads
waveforms to the longest waveform and labels to the longest label sequence, and
those two maxima can disagree by a frame on rounding alone, so an exact
equality assertion is too strict and fails on healthy batches.

Taking the minimum of the two is too lax in one direction, so the two
directions are treated differently:

  * more logits than labels  -> the extra columns are past every utterance's
                                alignment and carry no supervision. Harmless;
                                pad the labels with the ignore index.
  * more labels than logits  -> truncating discards supervised frames. Fatal.
  * a large gap either way   -> a frame-rate mismatch (fairseq_padding shifts
                                the count by ~1 frame per second). Fatal.

Run:
    python tests/test_probe_frame_alignment.py
    pytest -q tests/test_probe_frame_alignment.py
"""
import sys
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.probe_frozen_linear import (  # noqa: E402
    MAX_FRAME_SLACK, align_frames,
)

VOCAB = 43


def _batch(n_logits, n_labels, B=4):
    logits = torch.randn(B, n_logits, VOCAB)
    labels = torch.randint(0, VOCAB, (B, n_labels))
    return logits, labels


def test_equal_frame_counts_pass_through_unchanged():
    logits, labels = _batch(731, 731)
    out = align_frames(logits, labels, where='test')
    assert out.shape == labels.shape
    assert torch.equal(out, labels)


def test_one_extra_logit_frame_is_padded_with_the_ignore_index():
    """A case seen in practice: 732 logits vs 731 labels."""
    logits, labels = _batch(732, 731)
    out = align_frames(logits, labels, where='test')
    assert out.shape[1] == 732, 'labels were not padded up to the logit frames'
    assert (out[:, -1] == -100).all(), 'padding must be the ignore index'
    # The real labels are unchanged: nothing was dropped.
    assert torch.equal(out[:, :731], labels)


def test_padded_frames_contribute_nothing_to_the_loss():
    """Padding must not change the cross-entropy the probe optimises."""
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits, labels = _batch(300, 299, B=2)

    aligned = align_frames(logits, labels, where='test')
    padded_loss = F.cross_entropy(logits.reshape(-1, VOCAB),
                                  aligned.reshape(-1), ignore_index=-100)
    # Same computation with the extra frame simply removed.
    trimmed_loss = F.cross_entropy(logits[:, :299].reshape(-1, VOCAB),
                                   labels.reshape(-1), ignore_index=-100)
    assert abs(float(padded_loss) - float(trimmed_loss)) < 1e-6


def test_more_labels_than_logits_is_fatal():
    """Truncating here would discard real supervision, so it is an error."""
    logits, labels = _batch(730, 731)
    try:
        align_frames(logits, labels, where='probe training')
    except RuntimeError as exc:
        assert 'DISCARD' in str(exc)
        assert 'probe training' in str(exc)
        return
    raise AssertionError('labels longer than logits was allowed through')


def test_a_large_logit_excess_is_fatal():
    """~1 frame per second of audio is what a fairseq_padding mismatch looks like."""
    logits, labels = _batch(731 + MAX_FRAME_SLACK + 1, 731)
    try:
        align_frames(logits, labels, where='evaluate')
    except RuntimeError as exc:
        assert 'frame-rate mismatch' in str(exc)
        assert 'fairseq_padding' in str(exc)
        return
    raise AssertionError('a rate-sized gap was allowed through')


def test_the_slack_boundary_is_inclusive():
    """Exactly MAX_FRAME_SLACK is rounding; one more is a mismatch."""
    logits, labels = _batch(731 + MAX_FRAME_SLACK, 731)
    out = align_frames(logits, labels, where='test')
    assert out.shape[1] == 731 + MAX_FRAME_SLACK


# ----------------------------------------------------------------------------
def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n        {exc}")
        except Exception as exc:                     # noqa: BLE001
            failed += 1
            print(f"ERROR {name}\n        {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_main())
