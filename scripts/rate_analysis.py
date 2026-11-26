#!/usr/bin/env python3
"""
Streaming rate / entropy instrumentation for FSQ and RVQ evaluation.

Everything is accumulated as histograms: raw per-frame index tensors are never
concatenated, because a dev-clean evaluation covers 1.84M encoder positions
(zero-padded batches of 8 in manifest order, padding included) and FSQ-64d
would otherwise materialise a 64-wide code tensor over all of them.

Quantities produced
-------------------
FSQ (`FSQRateAccumulator`)
  * per-dimension marginal entropy H(dim_i), in bits, from a (d, max_L) count
    matrix, plus their sum. sum_i H(dim_i) upper-bounds the joint entropy and is
    the rate of "independent scalar channels".
  * nominal rate sum_i log2(L_i).
  * joint entropy over observed codes, plug-in and Miller-Madow corrected,
    reported alongside n_frames / n_unique_codes and an explicit saturation flag
    when the implicit codebook exceeds the frame count (where the plug-in
    estimator is bounded by log2(n_frames) and is therefore not the true joint
    entropy).

RVQ (`RVQRateAccumulator`)
  * per-layer entropy H_l and perplexity exp(H_l^nats) from an (L, K) count
    matrix, the entropy rate sum_l H_l, and the nominal rate L*log2(K).
  * joint entropy over the full L-tuple, which captures the inter-layer
    redundancy that the per-layer sum ignores.

Joint code identity
-------------------
Exact mixed-radix packing is used whenever the product of the radices fits in
int64 (for example RVQ 8L x 128 = 128^8 = 7.2e16 and FSQ [3]^32 = 1.9e15, but
not FSQ [3]^64 = 3.4e30). Otherwise a 64-bit multiplicative hash is used and
`joint_code_id` is reported as "hash64". Its coefficients are drawn from a
fixed-seed generator at every call, that is once per batch, so the same code
can receive different ids in different batches and the joint statistics are
then not code-level values.
"""
import math

import torch

LN2 = math.log(2.0)
INT64_MAX = 2 ** 63 - 1

# A joint entropy is called converged when doubling the sample from N/2 to N
# moves it by less than this fraction. Below it, the estimate is treated as a
# measurement; above it, as a sample-limited lower bound.
CONVERGENCE_THRESHOLD = 0.02


# ----------------------------------------------------------------------------
# Streaming exact counter over int64 symbol ids
# ----------------------------------------------------------------------------
class _SymbolCounter:
    """Accumulates exact counts over int64 symbols without storing raw frames.

    State is bounded by the number of distinct symbols observed, not by the
    number of frames: (keys, counts) are merged after every batch.
    """

    def __init__(self):
        self._keys = torch.empty(0, dtype=torch.int64)
        self._counts = torch.empty(0, dtype=torch.int64)
        self.n = 0

    def update(self, symbols: torch.Tensor):
        symbols = symbols.reshape(-1).to(torch.int64).cpu()
        self.n += symbols.numel()
        k, c = torch.unique(symbols, return_counts=True)
        if self._keys.numel() == 0:
            self._keys, self._counts = k, c
            return
        all_keys = torch.cat([self._keys, k])
        all_counts = torch.cat([self._counts, c])
        uniq, inverse = torch.unique(all_keys, return_inverse=True)
        merged = torch.zeros(uniq.numel(), dtype=torch.int64)
        merged.scatter_add_(0, inverse, all_counts)
        self._keys, self._counts = uniq, merged

    @property
    def n_unique(self) -> int:
        return int(self._keys.numel())

    def entropy_bits(self):
        """Return (plug-in H in bits, Miller-Madow corrected H in bits)."""
        if self.n == 0:
            return 0.0, 0.0
        p = self._counts.to(torch.float64) / float(self.n)
        h = float(-(p * torch.log2(p)).sum())
        # Miller-Madow: H_MM = H_plugin + (m - 1) / (2N), in nats.
        h_mm = h + (self.n_unique - 1) / (2.0 * self.n * LN2)
        return h, h_mm


def _mixed_radix_ids(codes: torch.Tensor, radices):
    """Pack (..., k) integer codes into unique int64 ids, or None on overflow.

    `codes[..., i]` must lie in [0, radices[i]).
    """
    total = 1
    for r in radices:
        total *= int(r)
        if total > INT64_MAX:
            return None
    strides, acc = [], 1
    for r in radices:
        strides.append(acc)
        acc *= int(r)
    stride_t = torch.tensor(strides, dtype=torch.int64, device=codes.device)
    return (codes.to(torch.int64) * stride_t).sum(dim=-1)


_HASH_GEN = torch.Generator().manual_seed(0x5EED_C0DE)


def _hash64_ids(codes: torch.Tensor, k: int):
    """64-bit multiplicative hash of (..., k) integer codes; the coefficients
    are drawn from a fixed-seed generator at every call."""
    coef = torch.randint(1, 2 ** 31, (k,), generator=_HASH_GEN,
                         dtype=torch.int64) * 2 + 1          # odd multipliers
    return (codes.to(torch.int64).cpu() * coef).sum(dim=-1)


class _JointCodeCounter:
    """Joint counter over a k-wide integer code, exact where possible."""

    # The convergence check keeps the joint ids so the N/8, N/4, N/2 prefixes
    # can be counted exactly, which is the one place this module trades its
    # frames-independent memory bound for accuracy. 8 bytes per frame: the
    # dev-clean evaluation has 1.84M positions, about 15 MB. Past the cap the check is
    # reported as null with a reason rather than silently approximated.
    MAX_BUFFERED_FRAMES = 20_000_000

    def __init__(self, radices):
        self.radices = [int(r) for r in radices]
        self.k = len(self.radices)
        self._counter = _SymbolCounter()
        probe = torch.zeros(1, self.k, dtype=torch.int64)
        self.exact = _mixed_radix_ids(probe, self.radices) is not None
        self.code_id = 'mixed_radix_exact' if self.exact else 'hash64'
        self._buf = []
        self._buffered = 0
        self._buffer_overflow = False

    def update(self, codes: torch.Tensor):
        ids = (_mixed_radix_ids(codes, self.radices) if self.exact
               else _hash64_ids(codes, self.k))
        self._counter.update(ids)
        if not self._buffer_overflow:
            flat = ids.reshape(-1).to(torch.int64).cpu()
            if self._buffered + flat.numel() > self.MAX_BUFFERED_FRAMES:
                self._buffer_overflow = True
                self._buf = []                      # release it
            else:
                self._buf.append(flat)
                self._buffered += flat.numel()

    def convergence(self):
        """Joint H over prefixes of the frame stream: is N large enough?

        A plug-in entropy estimate rises with sample size and only flattens
        once the sample covers the support. If H(N/2) -> H(N) still climbs,
        the reported joint entropy is a sample-limited lower bound, not an
        estimate of the truth, and should be reported as a lower bound, not as
        a measurement.
        """
        if self._buffer_overflow:
            return {'available': False,
                    'reason': f'frame count exceeded MAX_BUFFERED_FRAMES '
                              f'({self.MAX_BUFFERED_FRAMES:,}); the exact '
                              f'prefix check was not run.'}
        if not self._buf:
            return {'available': False, 'reason': 'no frames observed.'}

        ids = torch.cat(self._buf)
        n_total = int(ids.numel())
        points = []
        for frac, label in ((0.125, 'N/8'), (0.25, 'N/4'),
                            (0.5, 'N/2'), (1.0, 'N')):
            m = max(1, int(n_total * frac))
            c = _SymbolCounter()
            c.update(ids[:m])
            h, h_mm = c.entropy_bits()
            points.append({
                'fraction': label,
                'n_frames': m,
                'joint_entropy_bits': h,
                'joint_entropy_bits_miller_madow': h_mm,
                'n_unique_codes': c.n_unique,
            })

        h_half, h_full = points[2]['joint_entropy_bits'], points[3]['joint_entropy_bits']
        rel = ((h_full - h_half) / h_half) if h_half > 0 else 0.0
        return {
            'available': True,
            'points': points,
            'relative_increase_half_to_full': rel,
            'threshold': CONVERGENCE_THRESHOLD,
            'converged': bool(rel < CONVERGENCE_THRESHOLD),
        }

    def summary(self, nominal_codes):
        h, h_mm = self._counter.entropy_bits()
        n = self._counter.n
        saturated = bool(nominal_codes > n) if n else False
        return {
            'joint_entropy_bits': h,
            'joint_entropy_bits_miller_madow': h_mm,
            'joint_entropy_convergence': self.convergence(),
            'joint_code_id': self.code_id,
            'n_frames': n,
            'n_unique_codes': self._counter.n_unique,
            'nominal_codes': (float(nominal_codes)
                              if nominal_codes > INT64_MAX else int(nominal_codes)),
            'plugin_upper_bound_bits': math.log2(n) if n else 0.0,
            'estimator_saturated': saturated,
            'saturation_caveat': (
                'Implicit codebook exceeds the number of evaluated frames, so the '
                'plug-in joint entropy is bounded above by log2(n_frames) and '
                'UNDERSTATES the true joint entropy. Use the per-dimension '
                'marginal sum as the interpretable rate, and treat this value as '
                'a sample-limited lower bound.' if saturated else
                'Frame count exceeds the codebook size; the plug-in estimate is '
                'not sample-limited by codebook size.'
            ),
        }


# ----------------------------------------------------------------------------
# FSQ
# ----------------------------------------------------------------------------
class FSQRateAccumulator:
    """Per-dimension marginals + joint entropy for an FSQ model."""

    def __init__(self, levels):
        self.levels = [int(x) for x in levels]
        self.d = len(self.levels)
        self.max_L = max(self.levels)
        # (d, max_L) count matrix: the only per-dimension state.
        self.counts = torch.zeros(self.d, self.max_L, dtype=torch.int64)
        self.n_frames = 0
        self._joint = _JointCodeCounter(self.levels)

    def update(self, codes: torch.Tensor):
        """codes: (..., d) integer per-dimension levels in [0, L_i)."""
        flat = codes.reshape(-1, self.d).to(torch.int64).cpu()
        self.n_frames += flat.shape[0]
        for i in range(self.d):
            self.counts[i] += torch.bincount(flat[:, i], minlength=self.max_L)
        self._joint.update(flat)

    def summary(self):
        per_dim = []
        for i, L in enumerate(self.levels):
            c = self.counts[i, :L].to(torch.float64)
            tot = float(c.sum())
            if tot <= 0:
                per_dim.append({'dim': i, 'levels': L, 'entropy_bits': 0.0,
                                'n_used_levels': 0, 'nominal_bits': math.log2(L)})
                continue
            p = c / tot
            nz = p[p > 0]
            per_dim.append({
                'dim': i,
                'levels': L,
                'entropy_bits': float(-(nz * torch.log2(nz)).sum()),
                'n_used_levels': int((c > 0).sum()),
                'nominal_bits': math.log2(L),
            })

        nominal = sum(math.log2(L) for L in self.levels)
        marginal_sum = sum(x['entropy_bits'] for x in per_dim)
        nominal_codes = math.prod(self.levels)

        out = {
            'quantiser': 'fsq',
            'levels': self.levels,
            'd': self.d,
            'n_frames': self.n_frames,
            'per_dimension': per_dim,
            'marginal_entropy_bits_sum': marginal_sum,
            'nominal_rate_bits': nominal,
            'rate_efficiency': (marginal_sum / nominal) if nominal > 0 else 0.0,
        }
        out.update(self._joint.summary(nominal_codes))
        # Independence gap: how much of the marginal sum is inter-dimension
        # redundancy. Only meaningful when the joint estimate is not saturated.
        out['marginal_sum_minus_joint_bits'] = marginal_sum - out['joint_entropy_bits']
        return out


# ----------------------------------------------------------------------------
# RVQ
# ----------------------------------------------------------------------------
class RVQRateAccumulator:
    """Per-layer entropy/perplexity + joint entropy for a residual VQ stack."""

    def __init__(self, num_layers, num_codes):
        self.L = int(num_layers)
        self.K = int(num_codes)
        # (L, K) count matrix: the only per-layer state.
        self.counts = torch.zeros(self.L, self.K, dtype=torch.int64)
        self.n_frames = 0
        self._joint = _JointCodeCounter([self.K] * self.L)

    def update(self, all_indices):
        """all_indices: list of L tensors of shape (B, T) with values in [0, K)."""
        assert len(all_indices) == self.L, (
            f"expected {self.L} layers, got {len(all_indices)}")
        flats = [idx.reshape(-1).to(torch.int64).cpu() for idx in all_indices]
        self.n_frames += flats[0].numel()
        for l, f in enumerate(flats):
            self.counts[l] += torch.bincount(f, minlength=self.K)
        self._joint.update(torch.stack(flats, dim=-1))

    def summary(self):
        per_layer = []
        for l in range(self.L):
            c = self.counts[l].to(torch.float64)
            tot = float(c.sum())
            if tot <= 0:
                per_layer.append({'layer': l, 'entropy_bits': 0.0,
                                  'perplexity': 0.0, 'n_used_codes': 0,
                                  'n_dead_codes': self.K})
                continue
            p = c / tot
            nz = p[p > 0]
            h_bits = float(-(nz * torch.log2(nz)).sum())
            per_layer.append({
                'layer': l,
                'entropy_bits': h_bits,
                # Matches the training-log definition exp(-sum p ln p).
                'perplexity': float(2.0 ** h_bits),
                'n_used_codes': int((c > 0).sum()),
                'n_dead_codes': int(self.K - (c > 0).sum()),
                'utilisation_pct': float((c > 0).sum()) / self.K * 100.0,
            })

        nominal = self.L * math.log2(self.K)
        layer_sum = sum(x['entropy_bits'] for x in per_layer)
        nominal_codes = self.K ** self.L

        out = {
            'quantiser': 'rvq',
            'num_layers': self.L,
            'num_codes': self.K,
            'n_frames': self.n_frames,
            'per_layer': per_layer,
            'per_layer_perplexity': [x['perplexity'] for x in per_layer],
            'entropy_rate_bits_sum': layer_sum,
            'nominal_rate_bits': nominal,
            'rate_efficiency': (layer_sum / nominal) if nominal > 0 else 0.0,
        }
        out.update(self._joint.summary(nominal_codes))
        # Positive => layers carry redundant information (residual stages are
        # not independent). The per-layer sum alone would overstate the rate.
        out['layer_sum_minus_joint_bits'] = layer_sum - out['joint_entropy_bits']
        return out
