#!/usr/bin/env python3
"""
Tests for models/checkpoint_fingerprint.py.

Several load-bearing options change what a checkpoint's weights mean without
changing any parameter shape, so `load_state_dict(strict=False)` accepts a
mismatched checkpoint, reports zero missing and zero unexpected keys, and
produces a wrong number that looks right. `preserve_symmetry` is the extreme
case: it lives in non-persistent buffers, so it leaves no trace in the state
dict at all.

Obligations:
  1. MATCH:    an agreeing config passes and says what it checked.
  2. MISMATCH: any disagreement on a recorded field is fatal, including the
               shape-invisible ones.
  3. LEGACY:   a checkpoint with no fingerprint warns once and is allowed
               through, so that checkpoints written before the fingerprint
               remain loadable.
  4. LOADING:  a missing parameter key is fatal; an unexpected one is not.

Run:
    python tests/test_checkpoint_fingerprint.py
    pytest -q tests/test_checkpoint_fingerprint.py
"""
import logging
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.checkpoint_fingerprint import (  # noqa: E402
    FINGERPRINT_KEY, FingerprintMismatch, attach_fingerprint, build_fingerprint,
    load_state_dict_checked, reset_legacy_warning, validate_fingerprint,
)


class _Tiny(nn.Module):
    def __init__(self, d=4):
        super().__init__()
        self.a = nn.Linear(d, d)
        self.b = nn.Linear(d, d)


class _Args:
    """Stand-in for an argparse Namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ----------------------------------------------------------------------------
# 1. MATCH
# ----------------------------------------------------------------------------
def test_matching_config_passes_and_reports_what_it_checked():
    fp = build_fingerprint('ProtoSpeechFSQ',
                           _Args(fsq_levels=[8, 5, 5, 5], preserve_symmetry=False),
                           hidden_dim=512)
    ckpt = {FINGERPRINT_KEY: fp}
    result = validate_fingerprint(ckpt, dict(fp), checkpoint_path='t.pt')
    assert result['checked'] is True
    assert result['n_fields'] > 0


def test_levels_compare_by_value_not_by_spelling():
    """'8,5,5,5' from a CLI and [8,5,5,5] from a config are the same thing."""
    stored = build_fingerprint('ProtoSpeechFSQ', fsq_levels='8,5,5,5')
    rebuilt = build_fingerprint('ProtoSpeechFSQ', fsq_levels=[8, 5, 5, 5])
    validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt)   # must not raise


def test_absent_boolean_flag_equals_the_flag_being_off():
    """A trainer with no --preserve_symmetry must match an evaluator defaulting it False.

    store_true flags have exactly two states. A trainer that predates the flag
    records None; an evaluator whose argparse defaults it to False reconstructs
    False. Those describe the same model, and treating them as different would
    reject valid checkpoints of the identity and gated baselines.
    """
    trainer = build_fingerprint('ProtoSpeechIdentity', hidden_dim=512)     # no such arg
    evaluator = build_fingerprint('ProtoSpeechIdentity',
                                  preserve_symmetry=False, hidden_dim=512)
    assert trainer['preserve_symmetry'] is False
    validate_fingerprint({FINGERPRINT_KEY: trainer}, evaluator)   # must not raise
    # And the genuine difference is still caught.
    on = build_fingerprint('ProtoSpeechIdentity', preserve_symmetry=True,
                           hidden_dim=512)
    try:
        validate_fingerprint({FINGERPRINT_KEY: on}, evaluator)
    except FingerprintMismatch:
        return
    raise AssertionError('True vs False was allowed through')


def test_attach_fingerprint_writes_under_the_expected_key():
    save = {'model_state_dict': {}}
    attach_fingerprint(save, 'ProtoSpeechIdentity', hidden_dim=512)
    assert FINGERPRINT_KEY in save
    assert save[FINGERPRINT_KEY]['model_class'] == 'ProtoSpeechIdentity'
    assert save[FINGERPRINT_KEY]['hidden_dim'] == 512


def test_fingerprint_survives_a_torch_save_round_trip(tmp_path=None):
    """torch.save/load must not change how the fingerprint compares."""
    import tempfile
    fp = build_fingerprint('ProtoSpeechFSQ', fsq_levels=[3] * 16,
                           preserve_symmetry=True, hidden_dim=512)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'ckpt.pt'
        torch.save({'model_state_dict': {}, FINGERPRINT_KEY: fp}, path)
        loaded = torch.load(path, map_location='cpu', weights_only=False)
    validate_fingerprint(loaded, dict(fp))                     # must not raise


# ----------------------------------------------------------------------------
# 2. MISMATCH
# ----------------------------------------------------------------------------
def test_preserve_symmetry_mismatch_is_fatal():
    """The shape-invisible case: buffers are non-persistent, so only this catches it."""
    stored = build_fingerprint('ProtoSpeechFSQ', fsq_levels=[2] * 10,
                               preserve_symmetry=True)
    rebuilt = build_fingerprint('ProtoSpeechFSQ', fsq_levels=[2] * 10,
                                preserve_symmetry=False)
    try:
        validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt,
                             checkpoint_path='bin10d_sym/best.pt')
    except FingerprintMismatch as exc:
        assert 'preserve_symmetry' in str(exc)
        assert 'bin10d_sym/best.pt' in str(exc)
        return
    raise AssertionError('a preserve_symmetry mismatch was allowed through')


def test_fairseq_padding_mismatch_is_fatal():
    """Also shape-invisible: padding changes frame count, not weight shape."""
    stored = build_fingerprint('SSLEncoder', fairseq_padding=True)
    rebuilt = build_fingerprint('SSLEncoder', fairseq_padding=False)
    try:
        validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt)
    except FingerprintMismatch as exc:
        assert 'fairseq_padding' in str(exc)
        return
    raise AssertionError('a fairseq_padding mismatch was allowed through')


def test_extract_layers_mismatch_is_fatal():
    stored = build_fingerprint('ProtoSpeechGatedSupervised', extract_layers=[4, 5, 6])
    rebuilt = build_fingerprint('ProtoSpeechGatedSupervised', extract_layers=[2, 4, 6])
    try:
        validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt)
    except FingerprintMismatch as exc:
        assert 'extract_layers' in str(exc)
        return
    raise AssertionError('an extract_layers mismatch was allowed through')


def test_model_class_mismatch_is_fatal_but_can_be_relaxed():
    stored = build_fingerprint('ProtoSpeechFSQ', fsq_levels=[3] * 16)
    rebuilt = build_fingerprint('ProtoSpeechBPC', fsq_levels=[3] * 16)
    try:
        validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt)
        raise AssertionError('a model_class mismatch was allowed through')
    except FingerprintMismatch as exc:
        assert 'model_class' in str(exc)
    # strict_class=False is for evaluators that deliberately rebuild through a
    # compatible wrapper; the config fields are still checked.
    validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt, strict_class=False)


def test_every_mismatched_field_is_named_at_once():
    """One failure should tell you everything that is wrong, not just the first."""
    stored = build_fingerprint('SSLEncoder', fairseq_padding=True,
                               no_positional_conv=True, hidden_dim=512)
    rebuilt = build_fingerprint('SSLEncoder', fairseq_padding=False,
                                no_positional_conv=False, hidden_dim=256)
    try:
        validate_fingerprint({FINGERPRINT_KEY: stored}, rebuilt)
    except FingerprintMismatch as exc:
        msg = str(exc)
        for field in ('fairseq_padding', 'no_positional_conv', 'hidden_dim'):
            assert field in msg, f'{field} missing from the diff'
        return
    raise AssertionError('mismatches were allowed through')


# ----------------------------------------------------------------------------
# 3. LEGACY
# ----------------------------------------------------------------------------
def test_legacy_checkpoint_passes_with_a_warning():
    reset_legacy_warning()
    result = validate_fingerprint({'model_state_dict': {}},
                                  build_fingerprint('ProtoSpeechFSQ'))
    assert result['checked'] is False
    assert 'legacy' in result['reason']


def test_legacy_warning_is_emitted_once_per_process():
    """A sweep over many checkpoints should say this once, not once per checkpoint."""
    reset_legacy_warning()
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger('fingerprint-test')
    log.addHandler(_Collect())
    log.setLevel(logging.WARNING)
    try:
        for _ in range(5):
            validate_fingerprint({}, build_fingerprint('X'), log=log)
    finally:
        log.handlers.clear()
    assert len(records) == 1, f'expected 1 warning, got {len(records)}'


# ----------------------------------------------------------------------------
# 4. LOADING
# ----------------------------------------------------------------------------
def test_checked_load_accepts_an_exact_state_dict():
    model, other = _Tiny(), _Tiny()
    missing, unexpected = load_state_dict_checked(model, other.state_dict())
    assert missing == [] and unexpected == []


def test_checked_load_tolerates_unexpected_keys():
    """An eval model legitimately drops heads the trainer had."""
    model = _Tiny()
    sd = dict(model.state_dict())
    sd['extra_head.weight'] = torch.zeros(2, 2)
    missing, unexpected = load_state_dict_checked(model, sd)
    assert missing == []
    assert 'extra_head.weight' in unexpected


def test_checked_load_refuses_a_missing_parameter():
    """The whole point: a missing key means random weights, silently."""
    model = _Tiny()
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith('b.')}
    try:
        load_state_dict_checked(model, sd, source='ckpt.pt')
    except RuntimeError as exc:
        assert 'b.weight' in str(exc) or 'b.bias' in str(exc)
        assert 'random initialisation' in str(exc)
        return
    raise AssertionError('a missing parameter key was absorbed silently')


def test_checked_load_allows_explicitly_permitted_missing_keys():
    model = _Tiny()
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith('b.')}
    missing, _ = load_state_dict_checked(model, sd, allow_missing=('b.',))
    assert sorted(missing) == ['b.bias', 'b.weight']


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
