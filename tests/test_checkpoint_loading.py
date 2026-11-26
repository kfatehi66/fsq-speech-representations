#!/usr/bin/env python3
"""
Tests for models/checkpoint_loading.py.

The case this file covers: a binary symmetric-bound checkpoint rebuilt
without `preserve_symmetry`. Because that flag lives in non-persistent
buffers, the load reports zero missing and zero unexpected keys, the run
completes cleanly, and the reported accuracy is far below the model's true
accuracy, an error that looks like a finding.

Two things have to be true for that to happen:
  1. the model was constructed without the flag, and
  2. nothing checked the construction against the checkpoint's fingerprint.

`load_model_from_checkpoint` prevents both: it takes construction arguments
from the fingerprint, then validates. These tests pin that behaviour.

Run:
    python tests/test_checkpoint_loading.py
    pytest -q tests/test_checkpoint_loading.py
"""
import sys
import tempfile
import traceback
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.checkpoint_fingerprint import (  # noqa: E402
    FINGERPRINT_KEY, FingerprintMismatch, build_fingerprint,
    reset_legacy_warning,
)
from models.checkpoint_loading import (  # noqa: E402
    MODEL_TYPES, load_model_from_checkpoint,
)

BINARY = [2] * 10


def _write_fsq_ckpt(path, levels, preserve_symmetry):
    """A real FSQ checkpoint, built the way the trainers build one."""
    from scripts.train_fsq import ProtoSpeechFSQ
    from models.fsq import FSQ
    torch.manual_seed(0)
    m = ProtoSpeechFSQ(fsq_levels=levels)
    if preserve_symmetry:
        m.fsq = FSQ(input_dim=512, levels=levels, preserve_symmetry=True)
    torch.save({
        'model_state_dict': m.state_dict(),
        FINGERPRINT_KEY: build_fingerprint(
            m, fsq_levels=levels, preserve_symmetry=preserve_symmetry,
            hidden_dim=512),
        'step': 30000,
    }, path)
    return m


# ----------------------------------------------------------------------------
# 1. THE REGRESSION
# ----------------------------------------------------------------------------
def test_symmetric_checkpoint_is_rebuilt_with_the_symmetric_bound():
    """A symmetric-bound checkpoint must be rebuilt from its fingerprint."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'bin10d_sym.pt'
        _write_fsq_ckpt(p, BINARY, preserve_symmetry=True)

        # The caller says nothing about preserve_symmetry. The loader must
        # still get it right.
        model, info = load_model_from_checkpoint(p, model_type='fsq')

        assert info['config_used']['preserve_symmetry'] is True, (
            'the loader ignored the fingerprint and rebuilt under the '
            'asymmetric bound')
        assert model.fsq.preserve_symmetry is True
        assert info['missing_keys'] == [] and info['unexpected_keys'] == []


def test_caller_contradicting_the_fingerprint_is_fatal():
    """An override that disagrees with the file is ignored: the fingerprint wins and no error is raised."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'bin10d_sym.pt'
        _write_fsq_ckpt(p, BINARY, preserve_symmetry=True)
        try:
            load_model_from_checkpoint(
                p, model_type='fsq',
                overrides={'preserve_symmetry': False, 'fsq_levels': BINARY})
        except FingerprintMismatch:
            raise AssertionError(
                'the fingerprint should have won over the override, not '
                'conflicted with it')
        except Exception as exc:                     # noqa: BLE001
            raise AssertionError(f'unexpected failure: {type(exc).__name__}: {exc}')
        # The fingerprint outranks the override, so this loads and is symmetric.
        model, info = load_model_from_checkpoint(
            p, model_type='fsq', overrides={'preserve_symmetry': False})
        assert info['config_used']['preserve_symmetry'] is True


def test_asymmetric_checkpoint_stays_asymmetric():
    """The control: the same path must not turn a plain checkpoint symmetric."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'bin10d.pt'
        _write_fsq_ckpt(p, BINARY, preserve_symmetry=False)
        model, info = load_model_from_checkpoint(p, model_type='fsq')
        assert info['config_used']['preserve_symmetry'] is False
        assert model.fsq.preserve_symmetry is False


def test_the_two_bounds_produce_different_codes():
    """Why any of this matters: the flag changes the codes, not the shapes.

    If these agreed, mixing the bounds would be harmless and the loader would
    be bureaucracy. They do not agree.
    """
    from models.fsq import FSQ
    torch.manual_seed(0)
    z = torch.randn(2, 40, 512)
    asym = FSQ(input_dim=512, levels=BINARY, preserve_symmetry=False)
    sym = FSQ(input_dim=512, levels=BINARY, preserve_symmetry=True)
    sym.load_state_dict(asym.state_dict())          # identical parameters
    with torch.no_grad():
        ca = asym(z)['codes']
        cs = sym(z)['codes']
    assert ca.shape == cs.shape
    assert not torch.equal(ca, cs), (
        'the two bounds produced identical codes; the whole preserve_symmetry '
        'hazard would be a non-issue if that were true')


# ----------------------------------------------------------------------------
# 2. GENERAL CONTRACT
# ----------------------------------------------------------------------------
def test_levels_come_from_the_fingerprint_when_caller_is_silent():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'fsq8.pt'
        _write_fsq_ckpt(p, [3] * 8, preserve_symmetry=False)
        model, info = load_model_from_checkpoint(p, model_type='fsq')
        assert list(info['config_used']['fsq_levels']) == [3] * 8


def test_legacy_checkpoint_uses_overrides_and_warns_once():
    """No fingerprint -> caller config, one warning, still loads."""
    reset_legacy_warning()
    from scripts.train_fsq import ProtoSpeechFSQ
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'legacy.pt'
        torch.manual_seed(0)
        m = ProtoSpeechFSQ(fsq_levels=[3] * 8)
        torch.save({'model_state_dict': m.state_dict(), 'step': 1}, p)
        model, info = load_model_from_checkpoint(
            p, model_type='fsq', overrides={'fsq_levels': [3] * 8})
        assert info['had_fingerprint'] is False
        assert info['missing_keys'] == []


def test_missing_parameter_is_fatal():
    """A parameter absent from the file would otherwise stay randomly initialised."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / 'truncated.pt'
        m = _write_fsq_ckpt(p, [3] * 8, preserve_symmetry=False)
        ck = torch.load(p, map_location='cpu', weights_only=False)
        victim = next(k for k in ck['model_state_dict'] if 'fine_head' in k)
        del ck['model_state_dict'][victim]
        torch.save(ck, p)
        try:
            load_model_from_checkpoint(p, model_type='fsq')
        except RuntimeError as exc:
            assert 'random initialisation' in str(exc)
            return
        raise AssertionError('a missing parameter key was absorbed silently')


def test_unknown_model_type_is_rejected_by_name():
    try:
        load_model_from_checkpoint('/nonexistent.pt', model_type='not_a_model')
    except ValueError as exc:
        assert 'not_a_model' in str(exc)
        return
    raise AssertionError('an unknown model_type was accepted')


def test_every_advertised_model_type_has_a_builder():
    assert 'fsq' in MODEL_TYPES and 'ssl' in MODEL_TYPES
    assert len(MODEL_TYPES) >= 8


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
