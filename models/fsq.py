"""
Finite Scalar Quantization (FSQ), Mentzer et al. 2023
"Finite Scalar Quantization: VQ-VAE Made Simple"

Key idea: Instead of learning a codebook, bound each dimension to a finite
set of integer levels using tanh + round. The implicit codebook is the
Cartesian product of all per-dimension levels.

No codebook loss, no EMA updates, no commitment loss. Straight-through
estimator handles gradients through the round operation.
"""

import torch
import torch.nn as nn
import math
from typing import List


class FSQ(nn.Module):
    """Finite Scalar Quantization layer.

    Args:
        input_dim: Input feature dimension (D, typically 512).
        levels: List of ints, number of levels per dimension.
                e.g. [8, 5, 5, 5] gives 8*5*5*5 = 1000 implicit codes.
                The length of this list determines the low-dim bottleneck size d.
        preserve_symmetry: Opt-in symmetry-preserving bound (Parker et al. 2025,
                arXiv 2411.19842 §3.2). Default False. When False every code
                path below is byte-identical to the implementation without
                this option, for every level set, odd and even. See _bound /
                _quantize.

    Input:  (B, T, D)
    Output: dict matching ResidualVQ interface:
        - quantized:     (B, T, D), projected back to input dim
        - vq_loss:       0.0, as FSQ has no codebook loss
        - perplexities:  [scalar], codebook perplexity (single element list)
        - unique_codes:  [int], unique codes used (single element list)
        - all_indices:   [(B, T)], flat code indices (single element list)
        - residual_norm: 0.0, not applicable for FSQ
    """

    #: eps convention from the reference implementation (Mentzer et al. 2023,
    #: Appendix A.1). Applied to even-level dimensions only (see __init__).
    EPS = 1e-3

    def __init__(self, input_dim: int, levels: List[int],
                 preserve_symmetry: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.levels = levels
        self.d = len(levels)
        self.num_codes = math.prod(levels)
        self.preserve_symmetry = bool(preserve_symmetry)

        # Check if flat indices are representable in int64
        self._index_representable = self.num_codes <= (2**62)

        # Register levels as a buffer so they move to the correct device
        self.register_buffer('_levels', torch.tensor(levels, dtype=torch.float32))

        # Pre-compute strides for index calculation:
        # index = sum(z_q_i * prod(levels[i+1:]))
        if self._index_representable:
            strides = []
            for i in range(self.d):
                strides.append(math.prod(levels[i + 1:]))
            self.register_buffer('_strides', torch.tensor(strides, dtype=torch.long))
        else:
            # Strides overflow int64; flat indices will not be computed
            self.register_buffer('_strides', torch.ones(self.d, dtype=torch.long))

        # --- Bounding geometry (Mentzer et al. 2023, Appendix A.1) ---------
        # bound(z) = tanh(z + shift) * scale - offset,   then z_q = round(bound(z))
        #
        #   offset_i = 0.5 if L_i is even else 0.0
        #   scale_i  = (L_i - 1) / 2               for odd L_i
        #              (L_i - 1) * (1 + eps) / 2   for even L_i
        #   shift_i  = atanh(offset_i / scale_i)   (= 0 for odd L_i)
        #
        # The offset is what makes even L work. Without it, scale_i is a
        # half-integer and the open interval (-scale_i, +scale_i) spans only
        # L_i - 1 integer bins; at L_i = 2 it spans exactly one (z_q ≡ 0), so
        # binary FSQ can never emit more than a single code. With the offset
        # the reachable set is:
        #     odd  L: {-(L-1)/2, ..., +(L-1)/2}   -> L levels
        #     even L: {-L/2,     ..., +L/2 - 1}   -> L levels
        #
        # eps is applied to even dimensions only. Its sole purpose is to keep
        # offset/scale < 1 so that atanh is finite at L = 2 (where offset and
        # the un-eps'd scale are both 0.5). Odd dimensions need no eps, and
        # omitting it there keeps scale_i exactly (L_i - 1)/2, bit-identical to
        # models/fsq_legacy.py, which has no even-L offset, so the ternary results
        # do not change. See tests/test_fsq_bound.py.
        #
        # Index map: idx_i = z_q_i + (L_i // 2), pure integer arithmetic, so it
        # is a bijection onto [0, L_i - 1] for every L_i. (models/fsq_legacy.py
        # uses round(z_q_i + (L_i-1)/2), which for even L_i feeds half-integers
        # to round-half-to-even and collapses distinct levels onto shared
        # indices.)
        half_levels = torch.tensor([(l - 1) / 2.0 for l in levels], dtype=torch.float32)
        self.register_buffer('_half_levels', half_levels)

        self._all_odd = all(l % 2 == 1 for l in levels)

        scale, offset = [], []
        for l in levels:
            if l % 2 == 1:
                scale.append((l - 1) / 2.0)
                offset.append(0.0)
            else:
                scale.append((l - 1) * (1.0 + self.EPS) / 2.0)
                offset.append(0.5)
        scale_t = torch.tensor(scale, dtype=torch.float32)
        offset_t = torch.tensor(offset, dtype=torch.float32)

        # Derived geometry is recomputed from `levels` on every construction and
        # is marked non-persistent, so a checkpoint can never overwrite it.
        self.register_buffer('_scale', scale_t, persistent=False)
        self.register_buffer('_offset', offset_t, persistent=False)
        self.register_buffer('_shift', torch.atanh(offset_t / scale_t), persistent=False)
        self.register_buffer(
            '_index_offset',
            torch.tensor([l // 2 for l in levels], dtype=torch.long),
            persistent=False,
        )

        # --- preserve_symmetry geometry (unused when the flag is off) -------
        #
        # Parker et al. 2025 (arXiv 2411.19842 §3.2) replace Mentzer's asymmetric
        # bound with
        #     Q_L(x) = 2/(L-1) · floor[ (L-1)(tanh x + 1)/2 + 1/2 ] - 1
        # which lands on the normalised grid {-1, ..., +1}, L points, symmetric
        # about 0 for every L. Rescaling that grid by half_levels = (L-1)/2
        # onto this module's level grid, and using the identity
        #     floor(u + 1/2) == round(u)      (up to the tie convention)
        # it reduces exactly to:
        #
        #     "round  half_levels · tanh(z)  onto the symmetric level grid
        #      G_L = {-half_levels, -half_levels+1, ..., +half_levels}"
        #
        # G_L holds L points symmetric about 0. For odd L it is the integer set
        # {-(L-1)/2 ... +(L-1)/2}, identical to the flag-off odd path. For even L
        # it is a half-integer set: L=2 gives {-0.5, +0.5}, L=4 gives
        # {-1.5, -0.5, +0.5, +1.5}. This is the only difference. Mentzer's
        # even-L grid {-L/2 ... +L/2-1} is symmetric about -0.5 instead, which
        # moves the L=2 decision boundary out to z = -3.80; on G_2 the boundary
        # sits at z = 0 exactly.
        #
        # Consequences of the half-integer grid, both handled below:
        #   * torch.round() rounds to integers, so even dims need the shifted
        #     round  round(z+half)-half  (see _quantize).
        #   * z_q is no longer integral, so _codes_and_indices cannot use
        #     the integer _index_offset; it uses _half_levels instead, which is
        #     the exact left edge of G_L for both parities.
        #
        # The continuous part is written as the same expression as the all-odd
        # flag-off branch (half_levels * tanh(z)) rather than as their
        # algebraically equal (L-1)(tanh z + 1)/2 - half_levels, so that for odd
        # L the flag is bit-identical rather than merely equal to within
        # float32 rounding. See tests/test_fsq_bound.py.
        self.register_buffer(
            '_even_dim',
            torch.tensor([l % 2 == 0 for l in levels], dtype=torch.bool),
            persistent=False,
        )

        # Linear projections: D -> d (down) and d -> D (up)
        self.project_down = nn.Linear(input_dim, self.d)
        self.project_up = nn.Linear(self.d, input_dim)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound each dimension so that rounding yields exactly L_i levels.

            z_hat_i = tanh(z_i + shift_i) * scale_i - offset_i

        Reachable integer levels after rounding:
            odd  L_i: {-(L_i-1)/2, ..., +(L_i-1)/2}
            even L_i: {-L_i/2,     ..., +L_i/2 - 1}

        Under preserve_symmetry the reachable set is the symmetric grid G_L
        described in __init__ instead, for both parities.
        """
        if self.preserve_symmetry:
            # Symmetric bound. Textually identical to the all-odd branch below,
            # which is what makes odd-L flag-on bit-identical to flag-off; the
            # parities are separated in _quantize, not here.
            return self._half_levels * torch.tanh(z)
        if self._all_odd:
            # shift == 0 and offset == 0 for every dimension, so the general
            # expression below reduces to this. Written out explicitly so that
            # the ternary path is textually identical to models/fsq_legacy.py
            # and cannot drift. (Buffers are broadcast over (B, T, d).)
            return self._half_levels * torch.tanh(z)
        return torch.tanh(z + self._shift) * self._scale - self._offset

    def _quantize(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer with straight-through estimator."""
        if self.preserve_symmetry:
            # Even dims round onto the half-integer grid G_L: shifting by
            # half_levels maps G_L onto the integers [0, L-1], where round() is
            # exact, then shifts back. Odd dims keep the plain integer round
            # (the same expression as the flag-off line below), selected
            # per-dimension so that mixed level vectors are exact too.
            q_odd = torch.round(z_hat)
            q_even = torch.round(z_hat + self._half_levels) - self._half_levels
            q = torch.where(self._even_dim, q_even, q_odd)
            return z_hat + (q - z_hat).detach()
        z_q = z_hat + (torch.round(z_hat) - z_hat).detach()
        return z_q

    def _codes_and_indices(self, z_q: torch.Tensor):
        """Convert quantized vectors to per-dim codes and optional flat indices.

        Returns:
            codes: (B, T, d) per-dimension integer codes in [0, L-1]
            indices: (B, T) flat integer indices, or None if num_codes > int64
        """
        if self.preserve_symmetry:
            # z_q lies on G_L, whose left edge is -half_levels for both
            # parities, so z_q + half_levels is exactly [0, L-1]. (Both terms
            # are integers or half-integers of small magnitude, so the sum is
            # exact in float32 and round() only strips representation noise.)
            codes = torch.round(z_q + self._half_levels).long()
        else:
            # Shift to non-negative: [0, L-1]. z_q holds exact integer values,
            # so this is integer arithmetic throughout and the map is a
            # bijection.
            codes = torch.round(z_q).long() + self._index_offset
        # Clamp to valid range (safety; must never fire for a correct _bound)
        for i in range(self.d):
            codes[..., i] = codes[..., i].clamp(0, self.levels[i] - 1)
        # Flat index via strides (only when representable in int64)
        if self._index_representable:
            indices = (codes * self._strides).sum(dim=-1)
        else:
            indices = None
        return codes, indices

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (B, T, D) input features from transformer.

        Returns:
            dict matching ResidualVQ output interface.
        """
        B, T, D = z.shape

        # Project down: D -> d
        z_low = self.project_down(z)  # (B, T, d)

        # Bound to level ranges
        z_hat = self._bound(z_low)  # (B, T, d)

        # Quantize with STE
        z_q = self._quantize(z_hat)  # (B, T, d)

        # Compute per-dim codes and optional flat indices
        codes, indices = self._codes_and_indices(z_q)  # (B,T,d), (B,T) or None

        # Project back up: d -> D
        quantized = self.project_up(z_q)  # (B, T, D)

        # Codebook utilization metrics
        with torch.no_grad():
            if indices is not None:
                flat_indices = indices.reshape(-1)
                unique_vals, inv_counts = flat_indices.unique(return_counts=True)
                unique_codes = unique_vals.numel()
            else:
                # Use multi-dim unique for codebooks too large for int64
                codes_flat = codes.reshape(-1, self.d)
                _, _, inv_counts = torch.unique(codes_flat, dim=0,
                                                return_inverse=True, return_counts=True)
                unique_codes = inv_counts.numel()

            probs = inv_counts.float() / inv_counts.sum()
            perplexity = torch.exp(-torch.sum(probs * torch.log(probs + 1e-10)))

        return {
            'quantized': quantized,
            'vq_loss': torch.tensor(0.0, device=z.device),
            'perplexities': [perplexity],
            'unique_codes': [unique_codes],
            'all_indices': [indices],
            'codes': codes,
            'residual_norm': 0.0,
        }

    def get_codebook_utilization(self, indices=None, codes=None):
        """Compute detailed codebook utilization stats.

        Args:
            indices: (B, T) flat code indices from forward pass, or None.
            codes: (B, T, d) per-dimension codes from forward pass, or None.
                   Used when indices are None (codebook too large for int64).

        Returns:
            dict with utilization metrics.
        """
        if indices is not None:
            flat = indices.reshape(-1)
            total = flat.numel()
            unique_vals, inv_counts = flat.unique(return_counts=True)
            unique = unique_vals.numel()
        elif codes is not None:
            flat_codes = codes.reshape(-1, self.d)
            total = flat_codes.shape[0]
            _, _, inv_counts = torch.unique(flat_codes, dim=0,
                                            return_inverse=True, return_counts=True)
            unique = inv_counts.numel()
        else:
            raise ValueError("Must provide either indices or codes")

        # Entropy from used codes (unused codes contribute 0)
        probs = inv_counts.float() / total
        entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        max_entropy = math.log(self.num_codes) if self.num_codes > 1 else 0

        # Dead codes: full histogram for small codebooks, count-based for large
        if indices is not None and self.num_codes <= 100000:
            counts = torch.zeros(self.num_codes, device=indices.device)
            flat = indices.reshape(-1)
            counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
            avg_freq = total / self.num_codes
            num_dead = (counts < 0.01 * avg_freq).sum().item()
        else:
            num_dead = self.num_codes - unique
            counts = None

        return {
            'unique_codes': unique,
            'num_dead': num_dead,
            'utilization_pct': unique / self.num_codes * 100,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'entropy_ratio': entropy / max_entropy if max_entropy > 0 else 0,
            'counts': counts,
        }

    def extra_repr(self):
        return (f"levels={self.levels}, num_codes={self.num_codes}, "
                f"input_dim={self.input_dim}, bottleneck_dim={self.d}, "
                f"preserve_symmetry={self.preserve_symmetry}")
