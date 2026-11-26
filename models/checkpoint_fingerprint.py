#!/usr/bin/env python3
"""
A checkpoint's semantic identity, written at save time and checked at load time.

Every load site in this repository uses `load_state_dict(strict=False)`, and
several of the load-bearing configuration options do not change any parameter
shape. `preserve_symmetry` lives in non-persistent buffers; `fairseq_padding`
changes a conv's padding, not its weight; `extract_layers` picks which
Transformer layers get fused. A checkpoint trained under one setting therefore
loads cleanly under another, reports zero missing and zero unexpected keys, and
silently produces different codes and a different accuracy.

So the trainer records what it actually built, and the evaluator refuses to
proceed when its own reconstruction disagrees.

Written to be importable from both `models.` and `scripts.` without dragging in
torch model definitions: it only handles dicts.
"""
import logging

logger = logging.getLogger(__name__)

FINGERPRINT_KEY = 'config_fingerprint'

# The fields whose value changes what the weights mean. Adding a field here is
# cheap; leaving one out is how a silent mismatch gets through, so prefer to
# include anything the model reads at construction time.
FINGERPRINT_FIELDS = (
    'model_class',
    'fsq_levels',
    'preserve_symmetry',
    'fairseq_padding',
    'no_positional_conv',
    'extract_layers',
    'num_bpc_classes',
    'hidden_dim',
    'num_vq_layers',
    'num_codes',
)

# store_true flags. For these, "the trainer had no such argument" and "the flag
# was off" are the same state, so both normalise to False. Otherwise a
# trainer that predates a flag records None, an evaluator whose argparse
# defaults it to False reconstructs False, and the comparison fails on a
# difference that does not exist.
BOOLEAN_FIELDS = (
    'preserve_symmetry',
    'fairseq_padding',
    'no_positional_conv',
)

# Emitted once per process, not once per checkpoint: a sweep over many
# pre-fingerprint checkpoints should say so once.
_legacy_warned = False


def _norm(value):
    """Normalise for comparison: sequences to tuples, ints stay ints.

    `--fsq_levels 3,3,3` arrives as a string in one place and a list in
    another; `extract_layers` likewise. Compare the parsed value, not the
    spelling.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace(',', ' ').split() if p]
        if parts and all(p.lstrip('-').isdigit() for p in parts):
            return tuple(int(p) for p in parts)
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    if isinstance(value, bool):
        return bool(value)
    return value


def build_fingerprint(model_class, source=None, **overrides):
    """Assemble a fingerprint from an argparse Namespace / dict plus overrides.

    `model_class` may be a class, an instance, or a name. Anything absent from
    both `source` and `overrides` is recorded as None, which is meaningful:
    "this model does not have that knob".
    """
    if isinstance(model_class, str):
        name = model_class
    elif isinstance(model_class, type):
        name = model_class.__name__
    else:
        name = type(model_class).__name__

    src = {}
    if source is not None:
        src = dict(vars(source)) if hasattr(source, '__dict__') else dict(source)

    fp = {}
    for field in FINGERPRINT_FIELDS:
        if field == 'model_class':
            fp[field] = name
            continue
        raw = overrides[field] if field in overrides else src.get(field)
        fp[field] = bool(raw) if field in BOOLEAN_FIELDS else _norm(raw)
    # Tuples survive torch.save, but JSON round-trips turn them into lists;
    # store lists so both paths compare equal after _norm.
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in fp.items()}


def attach_fingerprint(save_dict, model_class, source=None, **overrides):
    """Add the fingerprint to a checkpoint dict, in place, and return it."""
    save_dict[FINGERPRINT_KEY] = build_fingerprint(model_class, source, **overrides)
    return save_dict


class FingerprintMismatch(RuntimeError):
    """The checkpoint was built under a different configuration than this one."""


def validate_fingerprint(ckpt, expected, *, checkpoint_path='<checkpoint>',
                         strict_class=True, log=None):
    """Compare a checkpoint's fingerprint against the config we just rebuilt.

    Checkpoints without a fingerprint warn once and pass, so that checkpoints
    written before the fingerprint existed remain loadable.

    Any disagreement on a recorded field raises FingerprintMismatch, because
    the alternative is a number that looks fine and is wrong.
    """
    global _legacy_warned
    lg = log or logger

    stored = (ckpt or {}).get(FINGERPRINT_KEY)
    if not stored:
        if not _legacy_warned:
            _legacy_warned = True
            lg.warning(
                'checkpoint has no %s: it predates the fingerprint. '
                'Its configuration cannot be verified against the one being '
                'reconstructed here; the load-bearing flags (preserve_symmetry, '
                'fairseq_padding, extract_layers, ...) are being taken on '
                'trust. Retrain to get a checked load. This warning is emitted '
                'once per process.', FINGERPRINT_KEY)
        return {'checked': False, 'reason': 'legacy checkpoint, no fingerprint'}

    diffs = []
    for field in FINGERPRINT_FIELDS:
        if field == 'model_class' and not strict_class:
            continue
        if field not in stored:
            continue                    # written by an older fingerprint version
        if field in BOOLEAN_FIELDS:
            want, got = bool(stored.get(field)), bool(expected.get(field))
        else:
            want, got = _norm(stored.get(field)), _norm(expected.get(field))
        if want != got:
            diffs.append((field, want, got))

    if diffs:
        detail = '\n'.join(
            f'    {f}: checkpoint has {w!r}, this run reconstructed {g!r}'
            for f, w, g in diffs)
        raise FingerprintMismatch(
            f'{checkpoint_path} was trained under a different configuration '
            f'than the one being reconstructed:\n{detail}\n'
            f'  These options change what the weights mean without changing '
            f'their shape, so load_state_dict would not have caught this. '
            f'Fix the arguments rather than removing this check.')

    lg.info('config fingerprint OK (%d fields checked against %s)',
            len([f for f in FINGERPRINT_FIELDS if f in stored]), checkpoint_path)
    return {'checked': True, 'n_fields': len(stored)}


def load_state_dict_checked(module, state_dict, *, source='<checkpoint>',
                            allow_missing=(), log=None):
    """`load_state_dict(strict=False)` that reports, and refuses to be silent.

    Unexpected keys are logged and tolerated: an eval model legitimately drops
    heads the trainer had. Missing keys are fatal: a parameter this model
    needs was not in the checkpoint, so it would be evaluated with random
    initialisation for that tensor.

    `allow_missing` is a tuple of prefixes for the rare deliberate case; every
    use of it should be commented at the call site.
    """
    lg = log or logger
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    lg.info('load_state_dict(%s): %d missing, %d unexpected',
            source, len(missing), len(unexpected))
    if unexpected:
        lg.info('  unexpected (present in checkpoint, unused here): %s%s',
                ', '.join(sorted(unexpected)[:8]),
                ' ...' if len(unexpected) > 8 else '')

    blocking = [k for k in missing
                if not any(k.startswith(p) for p in allow_missing)]
    if blocking:
        raise RuntimeError(
            f'{source}: {len(blocking)} parameter(s) required by this model '
            f'are absent from the checkpoint, so they would keep their random '
            f'initialisation: '
            f'{", ".join(sorted(blocking)[:10])}'
            f'{" ..." if len(blocking) > 10 else ""}. '
            f'This usually means the model was reconstructed with the wrong '
            f'architecture arguments.')
    if missing and not blocking:
        lg.warning('  %d missing key(s) explicitly allowed: %s',
                   len(missing), ', '.join(sorted(missing)[:8]))
    return missing, unexpected


def reset_legacy_warning():
    """Test hook: re-arm the once-per-process legacy warning."""
    global _legacy_warned
    _legacy_warned = False
