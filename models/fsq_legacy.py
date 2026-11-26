"""
Copy of models/fsq.py without the even-L offset, used by the regression tests only.

It is models/fsq.py as it was before the even-L bound was corrected. It lacks
the even-L offset: for even L it emits fewer than L levels (L=2 emits exactly
one), and its index map collides distinct levels onto shared indices.

Training and evaluation code do not import it. Its only consumers are
tests/test_fsq_bound.py and tests/test_fsq_e2e_regression.py, which use it to
show that the correction left the odd-L (ternary) path bit-identical. Its code
is kept unchanged, because those tests rely on it.

--- original docstring follows ---

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

    Input:  (B, T, D)
    Output: dict matching ResidualVQ interface:
        - quantized:     (B, T, D), projected back to input dim
        - vq_loss:       0.0, as FSQ has no codebook loss
        - perplexities:  [scalar], codebook perplexity (single element list)
        - unique_codes:  [int], unique codes used (single element list)
        - all_indices:   [(B, T)], flat code indices (single element list)
        - residual_norm: 0.0, not applicable for FSQ
    """

    def __init__(self, input_dim: int, levels: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.levels = levels
        self.d = len(levels)
        self.num_codes = math.prod(levels)

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

        # Pre-compute half-levels for offset: codes are 0..L-1, centered
        # z_hat = (L-1) * tanh(z), then round to integers in [-(L-1), (L-1)]
        # Shift to [0, L-1] for indexing: idx_i = (round(z_hat_i) + (L-1)) / 2
        # Following Mentzer: bound to [-floor(L/2), floor(L/2)]
        # then shift to [0, L-1] for index computation.
        half_levels = torch.tensor([(l - 1) / 2.0 for l in levels], dtype=torch.float32)
        self.register_buffer('_half_levels', half_levels)

        # Linear projections: D -> d (down) and d -> D (up)
        self.project_down = nn.Linear(input_dim, self.d)
        self.project_up = nn.Linear(self.d, input_dim)

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound each dimension to its level range using tanh.

        Maps z_i to [-(L_i-1)/2, (L_i-1)/2] via:
            z_hat_i = half_level_i * tanh(z_i)
        """
        # _half_levels shape: (d,), broadcast over (B, T, d)
        return self._half_levels * torch.tanh(z)

    def _quantize(self, z_hat: torch.Tensor) -> torch.Tensor:
        """Round to nearest integer with straight-through estimator."""
        z_q = z_hat + (torch.round(z_hat) - z_hat).detach()
        return z_q

    def _codes_and_indices(self, z_q: torch.Tensor):
        """Convert quantized vectors to per-dim codes and optional flat indices.

        Returns:
            codes: (B, T, d) per-dimension integer codes in [0, L-1]
            indices: (B, T) flat integer indices, or None if num_codes > int64
        """
        # Shift to non-negative: [0, L-1]
        codes = torch.round(z_q + self._half_levels).long()
        # Clamp to valid range (safety)
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
                f"input_dim={self.input_dim}, bottleneck_dim={self.d}")
