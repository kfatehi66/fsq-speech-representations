"""
Prototype Layer with EMA Updates (VQ-VAE style)

It counters the train/val collapse of the prototypes by:
1. Using EMA to update prototypes towards assigned embeddings
2. Resetting dead prototypes that aren't used
3. Using straight-through estimator for gradients
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class PrototypeLayerEMA(nn.Module):
    """
    Prototype layer with Exponential Moving Average updates.

    Unlike Sinkhorn (which only works during training), EMA updates
    ensure prototypes stay meaningful at inference time.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_prototypes: int = 50,
        temperature: float = 0.1,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_threshold: int = 100,  # Reset if unused for this many steps
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.ema_decay = ema_decay
        self.commitment_weight = commitment_weight
        self.dead_threshold = dead_threshold

        # Prototype embeddings (not updated by gradient, only by EMA)
        self.register_buffer('prototypes', torch.randn(num_prototypes, embed_dim))
        self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

        # EMA cluster sizes for tracking usage
        self.register_buffer('cluster_sizes', torch.zeros(num_prototypes))

        # Track when each prototype was last used
        self.register_buffer('last_used', torch.zeros(num_prototypes, dtype=torch.long))

        # Global step counter
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

        # Projection for hidden states
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        update_prototypes: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, D) encoder outputs
            update_prototypes: Whether to update prototypes with EMA

        Returns:
            Dictionary containing:
                - similarities: (B, T, P) similarity scores
                - assignments: (B, T, P) soft assignments
                - prototype_embeddings: (B, T, D) quantized embeddings
                - hard_assignments: (B, T) argmax indices
                - commitment_loss: scalar loss
        """
        B, T, D = hidden_states.shape

        # Project hidden states
        h = self.proj(hidden_states)  # (B, T, D)

        # Normalize for cosine similarity
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        # Compute similarities
        similarities = torch.matmul(h_norm, p_norm.T)  # (B, T, P)

        # Hard assignment (argmax)
        hard_indices = similarities.argmax(dim=-1)  # (B, T)

        # Soft assignment for gradients
        soft_assignments = F.softmax(similarities / self.temperature, dim=-1)

        # Get quantized embeddings using hard assignment
        # But use straight-through estimator for gradients
        quantized_hard = self.prototypes[hard_indices]  # (B, T, D)

        # Straight-through: forward uses hard, backward uses soft
        quantized = h + (quantized_hard - h).detach()

        # Commitment loss: encourage encoder outputs to commit to prototypes
        commitment_loss = F.mse_loss(h, quantized_hard.detach())

        # Update prototypes with EMA during training
        if self.training and update_prototypes:
            self._update_prototypes_ema(h.detach(), hard_indices)
            self.step += 1

            # Reset dead prototypes
            if self.step % 100 == 0:
                self._reset_dead_prototypes(h.detach())

        return {
            'similarities': similarities,
            'assignments': soft_assignments,
            'prototype_embeddings': quantized,
            'hard_assignments': hard_indices,
            'commitment_loss': commitment_loss,
        }

    def _update_prototypes_ema(
        self,
        hidden_states: torch.Tensor,
        assignments: torch.Tensor,
    ):
        """Update prototypes using Exponential Moving Average."""
        B, T, D = hidden_states.shape
        flat_h = hidden_states.view(-1, D)  # (B*T, D)
        flat_assign = assignments.view(-1)  # (B*T,)

        # Count assignments per prototype
        one_hot = F.one_hot(flat_assign, self.num_prototypes).float()  # (B*T, P)
        cluster_counts = one_hot.sum(dim=0)  # (P,)

        # Sum of assigned embeddings per prototype
        cluster_sums = torch.matmul(one_hot.T, flat_h)  # (P, D)

        # EMA update
        self.cluster_sizes.mul_(self.ema_decay).add_(
            cluster_counts, alpha=1 - self.ema_decay
        )

        # Update prototypes
        new_prototypes = cluster_sums / (cluster_counts.unsqueeze(-1) + 1e-6)
        mask = cluster_counts > 0

        if mask.any():
            self.prototypes[mask] = (
                self.ema_decay * self.prototypes[mask] +
                (1 - self.ema_decay) * new_prototypes[mask]
            )
            # Normalize prototypes
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

        # Update last_used for active prototypes
        self.last_used[mask] = self.step

    def _reset_dead_prototypes(self, hidden_states: torch.Tensor):
        """Reset prototypes that haven't been used recently."""
        dead_mask = (self.step - self.last_used) > self.dead_threshold
        num_dead = dead_mask.sum().item()

        if num_dead > 0:
            # Sample random embeddings from current batch
            flat_h = hidden_states.view(-1, self.embed_dim)
            random_indices = torch.randperm(flat_h.size(0))[:num_dead]

            # Reset dead prototypes to random embeddings
            self.prototypes.data[dead_mask] = F.normalize(
                flat_h[random_indices], dim=-1
            )
            self.last_used[dead_mask] = self.step
            self.cluster_sizes[dead_mask] = 0

            print(f"Reset {num_dead} dead prototypes at step {self.step.item()}")

    def get_effective_prototypes(self) -> float:
        """Compute effective number of prototypes based on usage."""
        # Use cluster sizes normalized
        probs = self.cluster_sizes / (self.cluster_sizes.sum() + 1e-6)
        probs = probs.clamp(min=1e-10)
        entropy = -(probs * probs.log()).sum()
        return entropy.exp().item()

    def get_prototype_stats(self) -> Dict[str, float]:
        """Get prototype diversity statistics (for compatibility)."""
        eff_proto = self.get_effective_prototypes()

        # Compute pairwise similarity between prototypes
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim_matrix = torch.matmul(p_norm, p_norm.T)

        # Exclude diagonal
        mask = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=sim_matrix.device)
        avg_sim = sim_matrix[mask].mean().item()
        max_sim = sim_matrix[mask].max().item()

        return {
            'effective_prototypes': eff_proto,
            'avg_similarity': avg_sim,  # Match expected key name
            'max_similarity': max_sim,
        }
