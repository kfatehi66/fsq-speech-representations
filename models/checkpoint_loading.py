#!/usr/bin/env python3
"""
The one way to turn a checkpoint into a model.

A script that reconstructs a model itself has a fresh chance to get a
load-bearing argument wrong. For example, a binary symmetric-bound checkpoint
rebuilt without `preserve_symmetry` loads with zero missing and zero
unexpected keys, runs cleanly, and reports a wrong accuracy. The fingerprint
check catches this, but only if the script calls it.

So construction is structural rather than left to each caller:

    model, info = load_model_from_checkpoint(path, model_type='fsq')

- the fingerprint is the source of truth for construction. Every load-bearing
  argument is read from it, so a caller cannot forget one. A caller override
  that disagrees with a recorded field is not used, and no error is raised.
- the fingerprint is then validated against what was actually built, which
  catches a builder that constructs something other than what the file
  records.
- loading goes through `load_state_dict_checked`, so a missing parameter is
  fatal rather than silently random.
- checkpoints without a fingerprint fall back to caller-supplied config and
  warn once per process, so that runs made before the fingerprint existed can
  still be loaded.

Adding a new architecture means adding one builder here, not a new load path.
"""
import logging

import torch

from models.checkpoint_fingerprint import (
    FINGERPRINT_KEY, build_fingerprint, load_state_dict_checked,
    validate_fingerprint,
)

logger = logging.getLogger(__name__)

# Defaults used only where neither the fingerprint nor the caller says
# otherwise. Kept explicit so a silent default can be traced.
_DEFAULTS = {
    'hidden_dim': 512,
    'num_vq_layers': 4,
    'num_codes': 128,
    'num_bpc_classes': 8,
    'extract_layers': [4, 5, 6],
    'preserve_symmetry': False,
    'fairseq_padding': False,
    'no_positional_conv': False,
}


def _cfg(fingerprint, ckpt_args, overrides, key):
    """Fingerprint, then the checkpoint's own args, then the caller, then default.

    The fingerprint is what the weights were actually built under, so it
    outranks anything a caller believes. A caller override that contradicts the
    fingerprint is not applied: the fingerprint value is used, and no error is
    raised.

    `ckpt_args` is the trainer's argparse Namespace where the trainer saved one
    (train_ssl_w2v2.py does). It ranks above the caller because it also comes
    from the file, though less structured than a fingerprint: a
    pre-fingerprint SSL checkpoint trained with --fairseq_padding says so in
    its own args, and a caller reconstructing from library defaults does not.
    """
    if fingerprint and fingerprint.get(key) is not None:
        return fingerprint[key]
    if ckpt_args and ckpt_args.get(key) is not None:
        return ckpt_args[key]
    if overrides and overrides.get(key) is not None:
        return overrides[key]
    return _DEFAULTS.get(key)


def _int_list(value):
    """Levels/layers arrive as a list, a tuple, or a "3,3,3" string.

    Trainer argparse writes the string form into `args`, which is what a legacy
    checkpoint's overrides carry; the fingerprint stores the parsed list. Both
    have to build the same model, and `[int(x) for x in "3,3,3"]` fails loudly
    on the comma rather than quietly, so parsing happens in one place.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return [int(p) for p in value.replace(',', ' ').split() if p]
    return [int(x) for x in value]


# ---------------------------------------------------------------------------
# Builders. One per architecture; imports are lazy so that loading an FSQ
# checkpoint does not drag in every trainer in the repo.
# ---------------------------------------------------------------------------
def _build_fsq(cfg, bpc_supervision):
    from scripts.train_fsq import ProtoSpeechFSQ
    levels = _int_list(cfg('fsq_levels'))
    hidden = cfg('hidden_dim')          # requested unconditionally: see note in
    symmetric = cfg('preserve_symmetry')  # load_model_from_checkpoint
    # Requested only when there is a BPC head, and the asymmetry is load-bearing
    # in both directions. A bpc-supervised checkpoint records the head width, so
    # not consulting it would read back as None and mismatch. A plain FSQ run
    # records num_bpc_classes: None (train_fsq_sym.py stamps the full field
    # list), so consulting it would reconstruct the default 8 and mismatch the
    # other way. The head only exists when bpc_supervision is on, so tie the
    # question to that.
    n_bpc = cfg('num_bpc_classes') if bpc_supervision else 8
    if levels is None:
        raise ValueError('fsq checkpoints need fsq_levels, from the fingerprint '
                         'or from the caller')
    model = ProtoSpeechFSQ(fsq_levels=levels, num_bpc_classes=n_bpc,
                           bpc_supervision=bpc_supervision)
    if symmetric:
        # The case this module exists for: preserve_symmetry lives in
        # non-persistent buffers, so a symmetric checkpoint loads cleanly under
        # the asymmetric bound and quietly produces different codes.
        from models.fsq import FSQ
        model.fsq = FSQ(input_dim=hidden, levels=levels,
                        preserve_symmetry=True)
    return model


def _build_rvq(cfg):
    from data.phoneme_vocab import VOCAB_SIZE
    from scripts.train_rvq_supervised import ProtoSpeechRVQ
    cfg('hidden_dim')
    return ProtoSpeechRVQ(num_vq_layers=cfg('num_vq_layers'),
                          num_codes=cfg('num_codes'),
                          num_phonemes=VOCAB_SIZE)


def _build_bpc(cfg):
    from scripts.train_rvq_bpc import ProtoSpeechBPC
    cfg('hidden_dim')
    return ProtoSpeechBPC(num_vq_layers=cfg('num_vq_layers'),
                          num_codes=cfg('num_codes'),
                          num_bpc_classes=cfg('num_bpc_classes'))


def _build_hierarchical(cfg):
    from data.phoneme_vocab import VOCAB_SIZE
    from scripts.train_rvq_hierarchical import ProtoSpeechHierarchical
    cfg('hidden_dim')
    return ProtoSpeechHierarchical(num_codes=cfg('num_codes'),
                                   num_bpc_classes=cfg('num_bpc_classes'),
                                   num_fine_classes=VOCAB_SIZE)


def _build_identity(cfg, bpc_supervision):
    from data.phoneme_vocab import VOCAB_SIZE
    from scripts.train_identity_baseline import ProtoSpeechIdentity
    cfg('hidden_dim')
    return ProtoSpeechIdentity(num_fine_classes=VOCAB_SIZE,
                               num_bpc_classes=cfg('num_bpc_classes'),
                               bpc_supervision=bpc_supervision)


def _build_gated(cfg):
    from data.phoneme_vocab import VOCAB_SIZE
    from scripts.train_gated_supervised import ProtoSpeechGatedSupervised
    cfg('hidden_dim')
    # extract_layers is passed explicitly rather than left to the constructor's
    # default, which would otherwise decide which layers get fused (the default
    # is the [4,5,6] that every run used).
    return ProtoSpeechGatedSupervised(
        num_phonemes=VOCAB_SIZE,
        extract_layers=_int_list(cfg('extract_layers')))


def _build_fusion(cfg):
    from data.phoneme_vocab import VOCAB_SIZE
    from scripts.train_fusion_phoneme_supervised import ProtoSpeechPhonemeSupervised
    return ProtoSpeechPhonemeSupervised(
        hidden_dim=cfg('hidden_dim'), num_layers=6, num_heads=8,
        num_phonemes=VOCAB_SIZE,
        extract_layers=_int_list(cfg('extract_layers')), dropout=0.1)


def _build_ssl(cfg, quantiser):
    from models.ssl_encoder import SSLEncoder
    cfg('hidden_dim')
    # Every quantiser field is requested regardless of branch. train_ssl_w2v2.py
    # fingerprints its whole argparse Namespace, so an rvq4l checkpoint records
    # the unused --fsq_levels default and an fsq16d one records the unused
    # --num_vq_layers / --num_codes. A builder that only consults its own
    # branch reports the others as None and mismatches a fingerprint that has
    # them, and the check would fire on fields the run never used.
    levels = _int_list(cfg('fsq_levels')) or [3] * 16
    n_layers, n_codes = cfg('num_vq_layers'), cfg('num_codes')
    quant_kw = {}
    if quantiser == 'fsq16d':
        quant_kw['fsq_levels'] = levels
    elif quantiser == 'rvq4l':
        quant_kw['num_vq_layers'] = n_layers
        quant_kw['num_codes'] = n_codes
    return SSLEncoder(quantiser=quantiser,
                      use_positional_conv=not cfg('no_positional_conv'),
                      fairseq_padding=cfg('fairseq_padding'),
                      **quant_kw)


_BUILDERS = {
    'fsq':          lambda c, k: _build_fsq(c, bpc_supervision=False),
    'fsq_bpc':      lambda c, k: _build_fsq(c, bpc_supervision=True),
    'rvq':          lambda c, k: _build_rvq(c),
    'bpc':          lambda c, k: _build_bpc(c),
    'hierarchical': lambda c, k: _build_hierarchical(c),
    'identity':     lambda c, k: _build_identity(c, bpc_supervision=k.get(
                                     'bpc_supervision', False)),
    'gated':        lambda c, k: _build_gated(c),
    'fusion':       lambda c, k: _build_fusion(c),
    'ssl':          lambda c, k: _build_ssl(c, k.get('quantiser', 'rvq4l')),
}

MODEL_TYPES = tuple(_BUILDERS)


def load_model_from_checkpoint(path, *, model_type, overrides=None,
                               device='cpu', strict_class=False, log=None,
                               allow_missing=(), **kwargs):
    """Build, validate and load a model from a checkpoint. The only load path.

    Returns (model, info) where info records what the fingerprint said, what
    was used, and the missing/unexpected key lists.

    `overrides` supplies configuration for checkpoints that carry no
    fingerprint. Where a fingerprint exists it wins: an override that
    contradicts a recorded field is ignored, and no error is raised.
    """
    lg = log or logger
    if model_type not in _BUILDERS:
        raise ValueError(f'unknown model_type {model_type!r}; '
                         f'expected one of {sorted(_BUILDERS)}')

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    fingerprint = ckpt.get(FINGERPRINT_KEY)

    # The trainer's own argparse record, where it saved one.
    ckpt_args = ckpt.get('args') or {}
    if not isinstance(ckpt_args, dict):
        ckpt_args = dict(vars(ckpt_args))

    # SSL checkpoints record the quantiser outside the fingerprint.
    if model_type == 'ssl' and 'quantiser' not in kwargs:
        kwargs['quantiser'] = ckpt.get('quantiser',
                                       ckpt_args.get('quantiser', 'rvq4l'))

    used = {}

    def cfg(key):
        used[key] = _cfg(fingerprint, ckpt_args, overrides, key)
        return used[key]

    model = _BUILDERS[model_type](cfg, kwargs)

    # Backstop. A field the builder never consulted would read back as None and
    # mismatch a fingerprint that records it, so fill it from the object that
    # was actually constructed, which is what the fingerprint should describe.
    if used.get('hidden_dim') is None and hasattr(model, 'hidden_dim'):
        used['hidden_dim'] = int(model.hidden_dim)

    # Validate what was built against what the file says it was trained as. The
    # expected fingerprint is assembled from `used`, so the check compares the
    # actual construction rather than the caller's intent.
    validate_fingerprint(
        ckpt,
        build_fingerprint(model, **{k: v for k, v in used.items()}),
        checkpoint_path=str(path), strict_class=strict_class, log=lg)

    missing, unexpected = load_state_dict_checked(
        model, ckpt['model_state_dict'], source=str(path),
        allow_missing=allow_missing, log=lg)

    model = model.to(device)
    return model, {
        'checkpoint': str(path),
        'model_type': model_type,
        'had_fingerprint': bool(fingerprint),
        'fingerprint': fingerprint,
        'config_used': used,
        'missing_keys': list(missing),
        'unexpected_keys': list(unexpected),
        'step': ckpt.get('step'),
        'raw': ckpt,
    }
