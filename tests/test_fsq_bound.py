#!/usr/bin/env python3
"""
Tests for the FSQ bounding function and code-index map (models/fsq.py).

Covers these obligations:

  1. LEVEL COUNT:      the bound must yield exactly L distinct integer levels
                       for every L, matching the reference table in this
                       file.
  2. BIJECTION:        level -> code-index must be a bijection onto [0, L-1]
                       for every dimension, including mixed-level vectors.
  3. ODD-L REGRESSION: for all-odd (ternary) configurations the bound and
                       index map must be bit-identical to models/fsq_legacy.py,
                       the implementation without the even-L offset. The
                       paper's multi-seed results depend on this.
  4. PRESERVE_SYMMETRY: the opt-in symmetric bound (Parker et al. 2025,
                       arXiv 2411.19842). With the flag off, obligations 1 to 3
                       hold unchanged; with the flag on, the level set must be
                       symmetric with the L=2 boundary at z ~= 0, and the bound
                       must be bit-identical to the flag-off bound for all-odd
                       configs.

Run either way:
    python tests/test_fsq_bound.py
    pytest -q tests/test_fsq_bound.py
"""
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.fsq import FSQ                      # noqa: E402  (with the even-L offset)
from models.fsq_legacy import FSQ as LegacyFSQ  # noqa: E402  (without the even-L offset)


# The reference sweep: z from -5 to +5 in steps of 0.25.
SWEEP = torch.arange(-5.0, 5.0 + 1e-6, 0.25)

# Reference reachable level sets. Odd L is symmetric about 0; even L is
# asymmetric (one extra negative level). That asymmetry is what the 0.5
# offset buys, and it is why L=2 yields {-1, 0} rather than {0}.
EXPECTED_LEVELS = {
    2: [-1, 0],
    3: [-1, 0, 1],
    4: [-2, -1, 0, 1],
    5: [-2, -1, 0, 1, 2],
    6: [-3, -2, -1, 0, 1, 2],
    7: [-3, -2, -1, 0, 1, 2, 3],
    8: [-4, -3, -2, -1, 0, 1, 2, 3],
    9: [-4, -3, -2, -1, 0, 1, 2, 3, 4],
}

ALL_L = sorted(EXPECTED_LEVELS)
MIXED_VECTORS = [[8, 5, 5, 5], [8, 8, 5, 5, 5], [2, 3], [4, 9, 6, 7], [3, 3, 3, 3]]


def _sweep_dim(module, level_index, z_values):
    """Push z_values through `module` on one dimension; return (z_q, codes)."""
    d = module.d
    z = torch.zeros(z_values.numel(), d)
    z[:, level_index] = z_values
    z_hat = module._bound(z)
    z_q = module._quantize(z_hat)
    codes, _ = module._codes_and_indices(z_q)
    return z_q[:, level_index], codes[:, level_index]


# ----------------------------------------------------------------------------
# 1. LEVEL COUNT
# ----------------------------------------------------------------------------
def test_distinct_levels_matches_reference_table():
    """Reference sweep: distinct output levels must equal L for every L."""
    for L in ALL_L:
        m = FSQ(input_dim=8, levels=[L]).eval()
        with torch.no_grad():
            z_q, _ = _sweep_dim(m, 0, SWEEP)
        got = sorted({int(v) for v in z_q.tolist()})
        assert got == EXPECTED_LEVELS[L], (
            f"L={L}: expected levels {EXPECTED_LEVELS[L]}, got {got}"
        )
        assert len(got) == L, f"L={L}: expected {L} distinct levels, got {len(got)}"


def test_saturation_does_not_escape_the_level_set():
    """Extreme inputs must not round outside the nominal level set."""
    extreme = torch.tensor([-1e4, -50.0, -20.0, 0.0, 20.0, 50.0, 1e4])
    for L in ALL_L:
        m = FSQ(input_dim=8, levels=[L]).eval()
        with torch.no_grad():
            z_q, codes = _sweep_dim(m, 0, extreme)
        got = sorted({int(v) for v in z_q.tolist()})
        assert set(got).issubset(set(EXPECTED_LEVELS[L])), (
            f"L={L}: saturation escaped the level set: {got}"
        )
        # The clamp in _codes_and_indices must never actually have to fire.
        assert int(codes.min()) >= 0 and int(codes.max()) <= L - 1


def test_binary_fsq_decision_boundary_is_reachable():
    """The main regression.

    Without the even-L offset (models/fsq_legacy.py), binary FSQ has no decision
    boundary anywhere on the real line: round(0.5*tanh(z)) == 0 for every finite
    z, so no encoder output can produce a second code. With the offset the
    boundary sits at a finite z, so the
    STE can actually move activations across it.

    Note this is a statement about reachability, not about initialisation. The
    boundary is at z ~= -3.80 because Mentzer's even-L bound is asymmetric, so
    a freshly-initialised projection (z_low ~ 0) starts saturated on one side
    and must learn to cross.
    """
    z = torch.linspace(-12.0, 12.0, 240001).unsqueeze(-1)

    new = FSQ(input_dim=8, levels=[2]).eval()
    old = LegacyFSQ(input_dim=8, levels=[2]).eval()
    with torch.no_grad():
        zq_new = torch.round(new._bound(z))[:, 0]
        zq_old = torch.round(old._bound(z))[:, 0]

    n_flips_new = int((zq_new[1:] != zq_new[:-1]).sum())
    n_flips_old = int((zq_old[1:] != zq_old[:-1]).sum())

    assert n_flips_old == 0, (
        "frozen legacy no longer reproduces the documented collapse: "
        "models/fsq_legacy.py must not be edited"
    )
    assert n_flips_new == 1, (
        f"expected exactly one L=2 decision boundary, found {n_flips_new}"
    )
    assert sorted({int(v) for v in zq_new.tolist()}) == [-1, 0]


def test_binary_fsq_reaches_full_codebook_when_driven():
    """[2]^10 must span its 1024 implicit codes given wide-enough activations."""
    torch.manual_seed(0)
    m = FSQ(input_dim=512, levels=[2] * 10).eval()
    with torch.no_grad():
        # Emulate a trained projection: wide activations, bias near the boundary.
        m.project_down.weight.normal_(0, 3.0 / math.sqrt(512))
        m.project_down.bias.fill_(-3.8)
        out = m(torch.randn(16, 256, 512))

    assert out['unique_codes'][0] > 900, (
        f"binary FSQ reached only {out['unique_codes'][0]}/1024 codes"
    )
    n_distinct = torch.unique(out['quantized'].reshape(-1, 512), dim=0).shape[0]
    assert n_distinct > 900, "quantiser output is near-constant"


# ----------------------------------------------------------------------------
# 2. BIJECTION
# ----------------------------------------------------------------------------
def _assert_bijective(module, dim_index, L):
    reachable = torch.tensor(EXPECTED_LEVELS[L], dtype=torch.float32)
    d = module.d
    z_q = torch.zeros(reachable.numel(), d)
    z_q[:, dim_index] = reachable
    codes, _ = module._codes_and_indices(z_q)
    got = codes[:, dim_index].tolist()

    assert len(set(got)) == len(got), (
        f"L={L} dim {dim_index}: index map is not injective: "
        f"levels {EXPECTED_LEVELS[L]} -> indices {got}"
    )
    assert sorted(got) == list(range(L)), (
        f"L={L} dim {dim_index}: image is {sorted(got)}, expected {list(range(L))}"
    )


def test_index_map_is_bijective_uniform():
    for L in ALL_L:
        m = FSQ(input_dim=8, levels=[L] * 3).eval()
        with torch.no_grad():
            for i in range(3):
                _assert_bijective(m, i, L)


def test_index_map_is_bijective_mixed():
    for levels in MIXED_VECTORS:
        m = FSQ(input_dim=64, levels=levels).eval()
        with torch.no_grad():
            for i, L in enumerate(levels):
                _assert_bijective(m, i, L)


def test_flat_index_is_bijective_over_the_full_product():
    """Enumerate the whole implicit codebook; flat indices must be a permutation."""
    for levels in ([2, 3], [4, 5], [8, 5, 5, 5], [2] * 6, [3] * 5):
        m = FSQ(input_dim=64, levels=levels).eval()
        grids = torch.meshgrid(
            *[torch.tensor(EXPECTED_LEVELS[L], dtype=torch.float32) for L in levels],
            indexing='ij',
        )
        z_q = torch.stack([g.reshape(-1) for g in grids], dim=-1)
        with torch.no_grad():
            _, indices = m._codes_and_indices(z_q)
        n = math.prod(levels)
        assert indices is not None
        assert sorted(indices.tolist()) == list(range(n)), (
            f"levels={levels}: flat indices are not a permutation of [0, {n})"
        )


def test_reachable_codebook_equals_nominal():
    """Drive forward() hard; reachable codes must equal prod(levels)."""
    torch.manual_seed(0)
    for levels in ([2] * 10, [8, 5, 5, 5], [8, 8, 5, 5, 5], [3] * 8):
        m = FSQ(input_dim=512, levels=levels).eval()
        with torch.no_grad():
            m.project_down.weight.normal_(0, 1.0)
            m.project_down.bias.normal_(0, 5.0)
            out = m(torch.randn(32, 400, 512) * 5)
        per_dim = [int(out['codes'][..., i].unique().numel()) for i in range(len(levels))]
        assert per_dim == list(levels), (
            f"levels={levels}: per-dim reachable {per_dim}, expected {list(levels)}"
        )


# ----------------------------------------------------------------------------
# 3. ODD-L REGRESSION (guards the ternary results of the paper)
# ----------------------------------------------------------------------------
ODD_CONFIGS = [
    [3] * 4, [3] * 5, [3] * 8, [3] * 16, [3] * 32, [3] * 64,
    [5] * 4, [7] * 3, [9] * 2, [3, 5, 7, 9], [5], [3],
]


def test_odd_levels_bound_and_index_bit_identical_to_legacy():
    """_bound / _quantize / index map of models/fsq.py == models/fsq_legacy.py, bit for bit."""
    torch.manual_seed(1234)
    for levels in ODD_CONFIGS:
        new = FSQ(input_dim=512, levels=list(levels)).eval()
        old = LegacyFSQ(input_dim=512, levels=list(levels)).eval()
        d = len(levels)

        for scale in (0.25, 1.0, 5.0, 50.0):
            z = torch.randn(4096, d) * scale
            with torch.no_grad():
                zh_new, zh_old = new._bound(z), old._bound(z)
                assert torch.equal(zh_new, zh_old), (
                    f"levels={levels} scale={scale}: _bound differs from legacy"
                )

                zq_new, zq_old = new._quantize(zh_new), old._quantize(zh_old)
                assert torch.equal(zq_new, zq_old), (
                    f"levels={levels} scale={scale}: z_q differs from legacy"
                )

                c_new, i_new = new._codes_and_indices(zq_new)
                c_old, i_old = old._codes_and_indices(zq_old)
                assert torch.equal(c_new, c_old), (
                    f"levels={levels} scale={scale}: codes differ from legacy"
                )
                if i_new is None or i_old is None:
                    assert i_new is None and i_old is None
                else:
                    assert torch.equal(i_new, i_old), (
                        f"levels={levels} scale={scale}: flat indices differ from legacy"
                    )


def test_odd_levels_full_forward_bit_identical_to_legacy():
    """Same weights in, identical quantised output / codes / metrics out."""
    torch.manual_seed(4321)
    for levels in ([3] * 16, [3] * 4, [3] * 8, [3, 5, 7, 9]):
        new = FSQ(input_dim=512, levels=list(levels)).eval()
        old = LegacyFSQ(input_dim=512, levels=list(levels)).eval()
        old.load_state_dict(new.state_dict(), strict=False)
        old.project_down.load_state_dict(new.project_down.state_dict())
        old.project_up.load_state_dict(new.project_up.state_dict())

        z = torch.randn(8, 200, 512) * 3
        with torch.no_grad():
            o_new, o_old = new(z), old(z)

        assert torch.equal(o_new['quantized'], o_old['quantized']), (
            f"levels={levels}: quantised output differs from legacy"
        )
        assert torch.equal(o_new['codes'], o_old['codes']), (
            f"levels={levels}: codes differ from legacy"
        )
        assert o_new['unique_codes'][0] == o_old['unique_codes'][0]
        assert torch.equal(o_new['perplexities'][0], o_old['perplexities'][0])


def test_odd_level_gradients_bit_identical_to_legacy():
    """STE gradient path must be unchanged for ternary configs."""
    torch.manual_seed(7)
    for levels in ([3] * 16, [5] * 4):
        new = FSQ(input_dim=512, levels=list(levels))
        old = LegacyFSQ(input_dim=512, levels=list(levels))
        old.load_state_dict(new.state_dict(), strict=False)
        old.project_down.load_state_dict(new.project_down.state_dict())
        old.project_up.load_state_dict(new.project_up.state_dict())

        z0 = torch.randn(4, 64, 512)
        zn = z0.clone().requires_grad_(True)
        zo = z0.clone().requires_grad_(True)
        new(zn)['quantized'].sum().backward()
        old(zo)['quantized'].sum().backward()
        assert torch.equal(zn.grad, zo.grad), (
            f"levels={levels}: input gradients differ from legacy"
        )


def test_even_levels_do_differ_from_legacy():
    """Sanity: the even-L correction must change even-L behaviour (guards against a correction with no effect)."""
    torch.manual_seed(11)
    for levels in ([2] * 10, [8, 5, 5, 5], [4] * 4):
        new = FSQ(input_dim=512, levels=list(levels)).eval()
        old = LegacyFSQ(input_dim=512, levels=list(levels)).eval()
        z = torch.randn(2048, len(levels)) * 3
        with torch.no_grad():
            c_new, _ = new._codes_and_indices(new._quantize(new._bound(z)))
            c_old, _ = old._codes_and_indices(old._quantize(old._bound(z)))
        assert not torch.equal(c_new, c_old), (
            f"levels={levels}: even-L output unchanged: the even-L correction did not take effect"
        )


# ----------------------------------------------------------------------------
# 4. PRESERVE_SYMMETRY (opt-in; Parker et al. 2025, arXiv 2411.19842)
# ----------------------------------------------------------------------------
# Under the flag every dimension lands on the symmetric grid
#     G_L = {-(L-1)/2, ..., +(L-1)/2}
# which is integral for odd L and half-integral for even L (L=2 -> {-0.5,+0.5}).
SYM_EXPECTED_LEVELS = {L: [k - (L - 1) / 2.0 for k in range(L)] for L in ALL_L}


def test_symmetry_flag_defaults_to_off():
    """The flag is opt-in: a bare FSQ(...) uses the asymmetric bound."""
    assert FSQ(input_dim=8, levels=[2]).preserve_symmetry is False
    assert FSQ(input_dim=8, levels=[3] * 4).preserve_symmetry is False


def test_symmetry_off_is_bit_identical_to_flagless_construction():
    """Explicit preserve_symmetry=False == omitting the argument, bit for bit.

    Guards against the flag's plumbing perturbing the default path.
    """
    torch.manual_seed(99)
    for levels in ([2] * 10, [8, 5, 5, 5], [3] * 16, [4, 9, 6, 7], [3] * 8):
        a = FSQ(input_dim=512, levels=list(levels)).eval()
        b = FSQ(input_dim=512, levels=list(levels), preserve_symmetry=False).eval()
        z = torch.randn(4096, len(levels)) * 3
        with torch.no_grad():
            zh_a, zh_b = a._bound(z), b._bound(z)
            assert torch.equal(zh_a, zh_b), f"levels={levels}: _bound differs"
            zq_a, zq_b = a._quantize(zh_a), b._quantize(zh_b)
            assert torch.equal(zq_a, zq_b), f"levels={levels}: z_q differs"
            c_a, i_a = a._codes_and_indices(zq_a)
            c_b, i_b = b._codes_and_indices(zq_b)
            assert torch.equal(c_a, c_b), f"levels={levels}: codes differ"
            assert torch.equal(i_a, i_b), f"levels={levels}: flat indices differ"


def test_symmetry_on_odd_levels_bit_identical_to_off():
    """Obligation 4, all-odd configs: the flag must be a no-op, bit for bit.

    Symmetry only changes even dimensions, so every published ternary result is
    unaffected by turning the flag on. Checked on _bound, z_q, codes, flat
    indices and the STE input gradient.
    """
    torch.manual_seed(2468)
    for levels in ODD_CONFIGS:
        off = FSQ(input_dim=512, levels=list(levels)).eval()
        on = FSQ(input_dim=512, levels=list(levels), preserve_symmetry=True).eval()
        d = len(levels)

        for scale in (0.25, 1.0, 5.0, 50.0):
            z = torch.randn(4096, d) * scale
            with torch.no_grad():
                zh_off, zh_on = off._bound(z), on._bound(z)
                assert torch.equal(zh_off, zh_on), (
                    f"levels={levels} scale={scale}: symmetric _bound differs")

                zq_off, zq_on = off._quantize(zh_off), on._quantize(zh_on)
                assert torch.equal(zq_off, zq_on), (
                    f"levels={levels} scale={scale}: symmetric z_q differs")

                c_off, i_off = off._codes_and_indices(zq_off)
                c_on, i_on = on._codes_and_indices(zq_on)
                assert torch.equal(c_off, c_on), (
                    f"levels={levels} scale={scale}: symmetric codes differ")
                if i_off is None or i_on is None:
                    assert i_off is None and i_on is None
                else:
                    assert torch.equal(i_off, i_on), (
                        f"levels={levels} scale={scale}: symmetric indices differ")

    # Gradients too: the STE path is what training sees.
    torch.manual_seed(13)
    for levels in ([3] * 16, [5] * 4, [3, 5, 7, 9]):
        off = FSQ(input_dim=512, levels=list(levels))
        on = FSQ(input_dim=512, levels=list(levels), preserve_symmetry=True)
        on.load_state_dict(off.state_dict(), strict=False)
        on.project_down.load_state_dict(off.project_down.state_dict())
        on.project_up.load_state_dict(off.project_up.state_dict())

        z0 = torch.randn(4, 64, 512)
        z_off = z0.clone().requires_grad_(True)
        z_on = z0.clone().requires_grad_(True)
        off(z_off)['quantized'].sum().backward()
        on(z_on)['quantized'].sum().backward()
        assert torch.equal(z_off.grad, z_on.grad), (
            f"levels={levels}: symmetric flag changed the odd-L gradient")


def test_symmetry_binary_has_exactly_two_symmetric_levels():
    """Flag on, L=2: exactly 2 levels, symmetric about 0."""
    m = FSQ(input_dim=8, levels=[2], preserve_symmetry=True).eval()
    with torch.no_grad():
        z_q, _ = _sweep_dim(m, 0, SWEEP)
    got = sorted({float(v) for v in z_q.tolist()})

    assert len(got) == 2, f"expected exactly 2 levels, got {got}"
    assert got == [-0.5, 0.5], f"expected {{-0.5, +0.5}}, got {got}"
    # Symmetric about zero: the set is closed under negation and sums to 0.
    assert sorted(-v for v in got) == got, f"level set is not symmetric: {got}"
    assert abs(sum(got)) < 1e-6, f"level set is not centred on 0: {got}"


def test_symmetry_binary_boundary_is_at_zero():
    """Flag on, L=2: exactly one decision boundary, and it sits at |z| < 0.1.

    This is the whole point of the flag. Under Mentzer's asymmetric bound the
    L=2 boundary is at z = -3.80, so a freshly-initialised projection
    (z_low ~ 0) starts fully saturated on one side; under the symmetric bound
    the boundary is at z = 0, where the initialisation already sits.
    """
    z = torch.linspace(-12.0, 12.0, 240001).unsqueeze(-1)
    m = FSQ(input_dim=8, levels=[2], preserve_symmetry=True).eval()
    with torch.no_grad():
        z_q = m._quantize(m._bound(z))[:, 0]

    flips = (z_q[1:] != z_q[:-1]).nonzero().reshape(-1)
    assert flips.numel() == 1, (
        f"expected exactly one L=2 decision boundary, found {flips.numel()}")

    boundary = float(z[flips[0] + 1, 0])
    assert abs(boundary) < 0.1, (
        f"symmetric L=2 boundary at z={boundary:.4f}, expected |z| < 0.1")

    # And it really is a different bound from the asymmetric one.
    off = FSQ(input_dim=8, levels=[2]).eval()
    with torch.no_grad():
        zq_off = off._quantize(off._bound(z))[:, 0]
    off_boundary = float(z[(zq_off[1:] != zq_off[:-1]).nonzero().reshape(-1)[0] + 1, 0])
    assert off_boundary < -3.0, (
        f"asymmetric L=2 boundary moved to {off_boundary:.4f}: flag-off changed")


def test_symmetry_levels_are_symmetric_for_every_L():
    """Flag on: the reachable level set is symmetric about 0 for every L."""
    for L in ALL_L:
        m = FSQ(input_dim=8, levels=[L], preserve_symmetry=True).eval()
        with torch.no_grad():
            z_q, _ = _sweep_dim(m, 0, SWEEP)
        got = sorted({float(v) for v in z_q.tolist()})
        assert got == SYM_EXPECTED_LEVELS[L], (
            f"L={L}: expected {SYM_EXPECTED_LEVELS[L]}, got {got}")
        assert len(got) == L, f"L={L}: expected {L} levels, got {len(got)}"
        assert sorted(-v for v in got) == got, f"L={L}: not symmetric: {got}"


def test_symmetry_index_map_is_bijective():
    """Flag on: level -> index must still be a bijection onto [0, L-1]."""
    for L in ALL_L:
        m = FSQ(input_dim=8, levels=[L] * 3, preserve_symmetry=True).eval()
        reachable = torch.tensor(SYM_EXPECTED_LEVELS[L], dtype=torch.float32)
        for i in range(3):
            z_q = torch.zeros(L, 3)
            z_q[:, i] = reachable
            with torch.no_grad():
                codes, _ = m._codes_and_indices(z_q)
            got = codes[:, i].tolist()
            assert len(set(got)) == L, (
                f"L={L} dim {i}: index map not injective: {reachable.tolist()} -> {got}")
            assert sorted(got) == list(range(L)), (
                f"L={L} dim {i}: image {sorted(got)}, expected {list(range(L))}")


def test_symmetry_mixed_vectors_with_binary_dims():
    """Flag on, mixed level vectors containing L=2 dims must behave per-dim.

    Even dims land on the half-integer grid, odd dims stay integral, and every
    dim reaches its full level count and a bijective index range.
    """
    torch.manual_seed(5)
    for levels in ([2, 3], [2, 3, 2, 5], [8, 5, 5, 5], [2] * 4 + [3] * 4,
                   [4, 9, 6, 7]):
        m = FSQ(input_dim=512, levels=list(levels), preserve_symmetry=True).eval()
        with torch.no_grad():
            m.project_down.weight.normal_(0, 1.0)
            m.project_down.bias.normal_(0, 2.0)
            out = m(torch.randn(32, 400, 512) * 5)

        z_q = m._quantize(m._bound(m.project_down(torch.randn(32, 400, 512) * 5)))
        for i, L in enumerate(levels):
            vals = sorted({float(v) for v in z_q[..., i].reshape(-1).tolist()})
            # Parity of the grid: even dims are half-integral, odd dims integral.
            is_half = all(abs(v - round(v)) > 0.4 for v in vals)
            if L % 2 == 0:
                assert is_half, f"levels={levels} dim {i} (L={L}): expected half-integers, got {vals}"
            else:
                assert not is_half, f"levels={levels} dim {i} (L={L}): expected integers, got {vals}"
            assert set(vals).issubset(set(SYM_EXPECTED_LEVELS[L])), (
                f"levels={levels} dim {i}: {vals} escapes G_{L}")

        per_dim = [int(out['codes'][..., i].unique().numel()) for i in range(len(levels))]
        assert per_dim == list(levels), (
            f"levels={levels}: per-dim reachable {per_dim}, expected {list(levels)}")
        assert int(out['codes'].min()) >= 0
        for i, L in enumerate(levels):
            assert int(out['codes'][..., i].max()) <= L - 1


def test_symmetry_binary_10d_reaches_full_codebook_from_default_init():
    """The substantive claim: [2]^10 spans its codebook without a learned bias.

    test_binary_fsq_reaches_full_codebook_when_driven needs bias = -3.8 to get
    the asymmetric bound off its saturated corner. The symmetric bound needs no
    such help: a default-initialised projection already straddles the boundary,
    which is why this variant is run as well.
    """
    torch.manual_seed(0)
    m = FSQ(input_dim=512, levels=[2] * 10, preserve_symmetry=True).eval()
    with torch.no_grad():
        out = m(torch.randn(16, 256, 512))   # default init, no bias surgery

    assert out['unique_codes'][0] > 900, (
        f"symmetric binary FSQ reached only {out['unique_codes'][0]}/1024 codes "
        f"from a default initialisation")

    # Contrast: the asymmetric bound collapses under the same conditions.
    torch.manual_seed(0)
    off = FSQ(input_dim=512, levels=[2] * 10).eval()
    with torch.no_grad():
        out_off = off(torch.randn(16, 256, 512))
    assert out_off['unique_codes'][0] < 10, (
        f"asymmetric [2]^10 reached {out_off['unique_codes'][0]} codes from "
        f"default init: the contrast this test documents no longer holds")


def test_symmetry_saturation_does_not_escape_the_level_set():
    """Flag on: extreme inputs must stay inside G_L."""
    extreme = torch.tensor([-1e4, -50.0, -20.0, 0.0, 20.0, 50.0, 1e4])
    for L in ALL_L:
        m = FSQ(input_dim=8, levels=[L], preserve_symmetry=True).eval()
        with torch.no_grad():
            z_q, codes = _sweep_dim(m, 0, extreme)
        vals = {float(v) for v in z_q.tolist()}
        assert vals.issubset(set(SYM_EXPECTED_LEVELS[L])), (
            f"L={L}: saturation escaped G_L: {sorted(vals)}")
        assert int(codes.min()) >= 0 and int(codes.max()) <= L - 1


def test_symmetry_flat_index_is_bijective_over_the_full_product():
    """Flag on: flat indices over the whole implicit codebook are a permutation."""
    for levels in ([2, 3], [4, 5], [8, 5, 5, 5], [2] * 6, [3] * 5):
        m = FSQ(input_dim=64, levels=levels, preserve_symmetry=True).eval()
        grids = torch.meshgrid(
            *[torch.tensor(SYM_EXPECTED_LEVELS[L], dtype=torch.float32) for L in levels],
            indexing='ij',
        )
        z_q = torch.stack([g.reshape(-1) for g in grids], dim=-1)
        with torch.no_grad():
            _, indices = m._codes_and_indices(z_q)
        n = math.prod(levels)
        assert indices is not None
        assert sorted(indices.tolist()) == list(range(n)), (
            f"levels={levels}: symmetric flat indices are not a permutation of [0, {n})")


def test_symmetry_gradients_flow_through_even_dims():
    """The STE must still pass gradient on the half-integer grid.

    A symmetric bound implemented with a bare floor() would zero the gradient;
    this asserts the straight-through path survives.
    """
    torch.manual_seed(3)
    m = FSQ(input_dim=512, levels=[2] * 10, preserve_symmetry=True)
    z = torch.randn(4, 64, 512).requires_grad_(True)
    m(z)['quantized'].sum().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert float(z.grad.abs().sum()) > 0, "symmetric STE produced a zero gradient"


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_main())
