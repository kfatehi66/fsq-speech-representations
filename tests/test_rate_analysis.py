#!/usr/bin/env python3
"""
Unit tests for scripts/rate_analysis.py.

Checks the entropy arithmetic against distributions with known closed-form
answers, and checks that the streaming accumulators give exactly the same
result as a single-shot computation over the concatenated data.

Run:
    python tests/test_rate_analysis.py
    pytest -q tests/test_rate_analysis.py
"""
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rate_analysis import (  # noqa: E402
    CONVERGENCE_THRESHOLD, FSQRateAccumulator, RVQRateAccumulator, _SymbolCounter,
)

TOL = 1e-9


def test_symbol_counter_uniform_entropy():
    """K equiprobable symbols -> H = log2(K) exactly."""
    for K in (2, 8, 128):
        c = _SymbolCounter()
        c.update(torch.arange(K).repeat(1000))
        h, h_mm = c.entropy_bits()
        assert abs(h - math.log2(K)) < 1e-6, f"K={K}: H={h}, want {math.log2(K)}"
        assert c.n_unique == K
        assert h_mm > h, "Miller-Madow correction must be positive"


def test_symbol_counter_degenerate_entropy():
    """A single symbol -> H = 0."""
    c = _SymbolCounter()
    c.update(torch.zeros(5000, dtype=torch.int64))
    h, _ = c.entropy_bits()
    assert abs(h) < TOL, f"expected 0 bits, got {h}"
    assert c.n_unique == 1


def test_symbol_counter_streaming_equals_single_shot():
    """Batched updates must equal one update over the concatenation."""
    torch.manual_seed(0)
    data = torch.randint(0, 50, (37, 211))
    streamed = _SymbolCounter()
    for row in data:
        streamed.update(row)
    single = _SymbolCounter()
    single.update(data)
    assert streamed.n == single.n
    assert streamed.n_unique == single.n_unique
    assert abs(streamed.entropy_bits()[0] - single.entropy_bits()[0]) < TOL


def test_miller_madow_formula():
    """H_MM = H_plugin + (m-1)/(2 N ln2)."""
    c = _SymbolCounter()
    c.update(torch.arange(16).repeat(64))
    h, h_mm = c.entropy_bits()
    expected = h + (16 - 1) / (2.0 * 16 * 64 * math.log(2.0))
    assert abs(h_mm - expected) < TOL


def test_fsq_uniform_independent_dims():
    """Independent uniform dims: sum_i H_i == joint H == d*log2(L)."""
    torch.manual_seed(1)
    levels = [3] * 5
    acc = FSQRateAccumulator(levels)
    codes = torch.randint(0, 3, (40000, 5))
    for chunk in codes.split(1000):
        acc.update(chunk)
    s = acc.summary()

    assert abs(s['nominal_rate_bits'] - 5 * math.log2(3)) < 1e-9
    assert abs(s['marginal_entropy_bits_sum'] - 5 * math.log2(3)) < 0.02
    # Independent dims -> marginal sum and joint entropy agree.
    assert abs(s['marginal_sum_minus_joint_bits']) < 0.05
    assert s['joint_code_id'] == 'mixed_radix_exact'
    assert s['n_frames'] == 40000
    assert not s['estimator_saturated'], "3^5=243 < 40000 frames"


def test_fsq_perfectly_correlated_dims():
    """Fully redundant dims: joint H = log2(L), marginal sum = d*log2(L)."""
    torch.manual_seed(2)
    levels = [4] * 3
    acc = FSQRateAccumulator(levels)
    base = torch.randint(0, 4, (20000, 1))
    acc.update(base.repeat(1, 3))
    s = acc.summary()
    assert abs(s['joint_entropy_bits'] - 2.0) < 0.02
    assert abs(s['marginal_entropy_bits_sum'] - 6.0) < 0.02
    assert abs(s['marginal_sum_minus_joint_bits'] - 4.0) < 0.05


def test_fsq_collapsed_dim_reports_zero_bits():
    """A dim stuck on one level contributes 0 bits (the [2]^10 case of models/fsq_legacy.py)."""
    acc = FSQRateAccumulator([2] * 4)
    acc.update(torch.zeros(1000, 4, dtype=torch.int64))
    s = acc.summary()
    assert abs(s['marginal_entropy_bits_sum']) < TOL
    assert all(x['n_used_levels'] == 1 for x in s['per_dimension'])
    assert s['n_unique_codes'] == 1
    assert abs(s['rate_efficiency']) < TOL


def test_fsq_saturation_flag():
    """Codebook > n_frames must set the saturation flag and the bound."""
    acc = FSQRateAccumulator([3] * 16)          # 43,046,721 codes
    acc.update(torch.randint(0, 3, (5000, 16)))
    s = acc.summary()
    assert s['estimator_saturated'] is True
    assert s['joint_entropy_bits'] <= s['plugin_upper_bound_bits'] + TOL
    assert abs(s['plugin_upper_bound_bits'] - math.log2(5000)) < TOL
    assert 'UNDERSTATES' in s['saturation_caveat']


def test_fsq_hash_fallback_for_wide_codebooks():
    """d=64 ternary overflows int64 -> hash64 path, still finite entropy."""
    acc = FSQRateAccumulator([3] * 64)
    acc.update(torch.randint(0, 3, (2000, 64)))
    s = acc.summary()
    assert s['joint_code_id'] == 'hash64'
    assert s['n_unique_codes'] > 1
    assert abs(s['marginal_entropy_bits_sum'] - 64 * math.log2(3)) < 0.5


def test_rvq_uniform_layers():
    """L layers of K equiprobable codes: H_l = log2 K, perplexity = K."""
    torch.manual_seed(3)
    acc = RVQRateAccumulator(num_layers=4, num_codes=128)
    for _ in range(20):
        acc.update([torch.randint(0, 128, (8, 300)) for _ in range(4)])
    s = acc.summary()
    assert abs(s['nominal_rate_bits'] - 4 * 7) < TOL
    for layer in s['per_layer']:
        assert abs(layer['entropy_bits'] - 7.0) < 0.05
        assert abs(layer['perplexity'] - 128) < 5
        assert layer['n_used_codes'] == 128
    assert abs(s['entropy_rate_bits_sum'] - 28.0) < 0.2
    assert s['joint_code_id'] == 'mixed_radix_exact'


def test_rvq_perplexity_matches_training_log_definition():
    """perplexity == exp(-sum p ln p), the formula of VQLayer.forward in train_rvq_bpc.py."""
    torch.manual_seed(4)
    idx = torch.randint(0, 128, (4000,))
    # Skew it so perplexity is well below K.
    idx[:3000] = torch.randint(0, 12, (3000,))
    acc = RVQRateAccumulator(num_layers=1, num_codes=128)
    acc.update([idx.reshape(1, -1)])
    got = acc.summary()['per_layer'][0]['perplexity']

    counts = torch.bincount(idx, minlength=128).float()
    p = counts / counts.sum()
    want = float(torch.exp(-(p * torch.log(p + 1e-10)).sum()))
    assert abs(got - want) / want < 1e-3, f"got {got}, want {want}"


def test_rvq_joint_captures_inter_layer_redundancy():
    """Duplicated layers: joint H = H_1, so layer_sum - joint = (L-1)*H_1."""
    torch.manual_seed(5)
    base = torch.randint(0, 64, (8, 500))
    acc = RVQRateAccumulator(num_layers=3, num_codes=64)
    acc.update([base, base.clone(), base.clone()])
    s = acc.summary()
    h1 = s['per_layer'][0]['entropy_bits']
    assert abs(s['joint_entropy_bits'] - h1) < 0.05
    assert abs(s['layer_sum_minus_joint_bits'] - 2 * h1) < 0.1


def test_rvq_dead_codes_counted():
    acc = RVQRateAccumulator(num_layers=2, num_codes=128)
    acc.update([torch.randint(0, 24, (8, 400)), torch.randint(0, 49, (8, 400))])
    s = acc.summary()
    assert s['per_layer'][0]['n_used_codes'] == 24
    assert s['per_layer'][0]['n_dead_codes'] == 104
    assert s['per_layer'][1]['n_used_codes'] == 49
    assert s['per_layer'][1]['n_dead_codes'] == 79


def test_rvq_streaming_equals_single_shot():
    torch.manual_seed(6)
    layers = [torch.randint(0, 128, (16, 400)) for _ in range(4)]
    one = RVQRateAccumulator(4, 128)
    one.update(layers)
    many = RVQRateAccumulator(4, 128)
    for b in range(16):
        many.update([lyr[b:b + 1] for lyr in layers])
    a, b = one.summary(), many.summary()
    assert a['n_frames'] == b['n_frames']
    assert abs(a['entropy_rate_bits_sum'] - b['entropy_rate_bits_sum']) < 1e-9
    assert abs(a['joint_entropy_bits'] - b['joint_entropy_bits']) < 1e-9


# ----------------------------------------------------------------------------
# Subsample convergence
# ----------------------------------------------------------------------------
def test_convergence_reports_the_four_prefixes_in_order():
    """N/8, N/4, N/2, N, each an exact prefix count."""
    torch.manual_seed(0)
    acc = FSQRateAccumulator([3, 3, 3])
    for _ in range(10):
        acc.update(torch.randint(0, 3, (4000, 3)))
    conv = acc.summary()['joint_entropy_convergence']

    assert conv['available'] is True
    assert [p['fraction'] for p in conv['points']] == ['N/8', 'N/4', 'N/2', 'N']
    n_total = conv['points'][-1]['n_frames']
    assert n_total == 40000
    for p, frac in zip(conv['points'], (0.125, 0.25, 0.5, 1.0)):
        assert p['n_frames'] == int(n_total * frac)
    # A prefix cannot see more distinct symbols than the whole stream.
    uniq = [p['n_unique_codes'] for p in conv['points']]
    assert uniq == sorted(uniq)


def test_convergence_true_when_the_sample_covers_the_support():
    """27 codes over 40k frames: doubling the sample must not move H."""
    torch.manual_seed(1)
    acc = FSQRateAccumulator([3, 3, 3])
    for _ in range(10):
        acc.update(torch.randint(0, 3, (4000, 3)))
    conv = acc.summary()['joint_entropy_convergence']
    assert conv['converged'] is True
    assert conv['relative_increase_half_to_full'] < CONVERGENCE_THRESHOLD


def test_convergence_false_when_the_codebook_dwarfs_the_sample():
    """3^16 codes over 8k frames: H is still climbing, so not converged.

    This is the case the flag exists for: the reported joint entropy is a
    sample-limited lower bound, and calling it a measurement would overstate it.
    """
    torch.manual_seed(2)
    acc = FSQRateAccumulator([3] * 16)
    for _ in range(4):
        acc.update(torch.randint(0, 3, (2000, 16)))
    summary = acc.summary()
    conv = summary['joint_entropy_convergence']
    assert conv['converged'] is False
    assert conv['relative_increase_half_to_full'] > CONVERGENCE_THRESHOLD
    # And it agrees with the saturation flag, which is the other symptom.
    assert summary['estimator_saturated'] is True


def test_convergence_final_point_matches_the_streaming_estimate():
    """The N point must equal the entropy the accumulator already reports."""
    torch.manual_seed(3)
    acc = RVQRateAccumulator(2, 128)
    for _ in range(8):
        acc.update([torch.randint(0, 128, (1500,)), torch.randint(0, 128, (1500,))])
    summary = acc.summary()
    conv = summary['joint_entropy_convergence']
    assert abs(conv['points'][-1]['joint_entropy_bits']
               - summary['joint_entropy_bits']) < TOL
    assert conv['points'][-1]['n_frames'] == summary['n_frames']


def test_convergence_unavailable_past_the_buffer_cap():
    """Past the cap the check reports itself unavailable, never a guess."""
    acc = FSQRateAccumulator([2, 2])
    acc._joint.MAX_BUFFERED_FRAMES = 100          # instance override
    acc.update(torch.randint(0, 2, (500, 2)))
    conv = acc.summary()['joint_entropy_convergence']
    assert conv['available'] is False
    assert 'MAX_BUFFERED_FRAMES' in conv['reason']
    # The headline entropy is still computed. Only the extra check is skipped.
    assert acc.summary()['joint_entropy_bits'] > 0


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
