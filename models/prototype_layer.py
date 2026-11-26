"""
Prototype layer of the ProtoSpeech model.

This module implements learnable phonetic prototypes for speech SSL models.

Key features:
1. Learnable prototypes initialized on unit hypersphere
2. Cosine similarity computation for prototype matching
3. Temperature-scaled softmax for soft assignments
4. Prototype-weighted embeddings for downstream tasks
"""

import math
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeLayer(nn.Module):
    """
    Learnable Prototype Layer for interpretable speech representations.

    Learns P prototype vectors that represent phonetic units. Each frame
    is assigned to prototypes via soft assignment, enabling:
    - "This sounds like prototype X" explanations
    - Prototype sonification (convert prototype to audio)
    - Prototype-phoneme alignment analysis

    Args:
        embed_dim: Dimension of input embeddings (default: 768)
        num_prototypes: Number of learnable prototypes (default: 100)
        temperature: Temperature for softmax (lower = sharper assignments)
        use_projection: Whether to project inputs before similarity
        dropout: Dropout probability for prototype embeddings
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_prototypes: int = 100,
        temperature: float = 0.1,
        use_projection: bool = True,
        dropout: float = 0.0,
        use_sinkhorn: bool = False,
        sinkhorn_iterations: int = 3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
        self.use_projection = use_projection
        self.use_sinkhorn = use_sinkhorn

        # Learnable prototypes - initialized on unit hypersphere
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, embed_dim))
        self._init_prototypes()

        # Optional projection layer
        if use_projection:
            self.projection = nn.Linear(embed_dim, embed_dim, bias=False)
        else:
            self.projection = nn.Identity()

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Sinkhorn normalization for balanced assignments
        if use_sinkhorn:
            from losses.diversity_loss import SinkhornKnopp
            self.sinkhorn = SinkhornKnopp(num_iterations=sinkhorn_iterations)

    def _init_prototypes(self):
        """Initialize prototypes uniformly on unit hypersphere.

        This ensures prototypes start diverse (low pairwise similarity).
        """
        # Xavier uniform initialization
        nn.init.xavier_uniform_(self.prototypes)

        # Normalize to unit hypersphere
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_distances: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prototype similarities and assignments.

        Args:
            hidden_states: Transformer outputs (B, T, D)
            return_distances: Also return Euclidean distances

        Returns:
            Dictionary containing:
                - similarities: (B, T, P) cosine similarity to each prototype
                - assignments: (B, T, P) soft assignment probabilities
                - prototype_embeddings: (B, T, D) prototype-weighted features
                - hard_assignments: (B, T) argmax prototype indices
                - distances: (B, T, P) Euclidean distances (if requested)
        """
        B, T, D = hidden_states.shape

        # Project hidden states (optional)
        h = self.projection(hidden_states)  # (B, T, D)
        h = self.dropout(h)

        # Normalize for cosine similarity
        h_norm = F.normalize(h, dim=-1)  # (B, T, D)
        p_norm = F.normalize(self.prototypes, dim=-1)  # (P, D)

        # Compute cosine similarities
        # (B, T, D) @ (D, P) -> (B, T, P)
        similarities = torch.matmul(h_norm, p_norm.t())

        # Temperature-scaled softmax for soft assignments
        # Lower temperature = sharper (more confident) assignments
        if self.training and self.use_sinkhorn:
            # Use Sinkhorn for balanced assignments during training
            B, T, P = similarities.shape
            flat_sim = similarities.view(B * T, P)
            balanced = self.sinkhorn(flat_sim)
            assignments = balanced.view(B, T, P)
        else:
            assignments = F.softmax(similarities / self.temperature, dim=-1)  # (B, T, P)

        # Prototype-weighted embedding (for downstream tasks)
        # (B, T, P) @ (P, D) -> (B, T, D)
        prototype_embeddings = torch.matmul(assignments, self.prototypes)

        # Hard assignments (argmax)
        hard_assignments = similarities.argmax(dim=-1)  # (B, T)

        outputs = {
            'similarities': similarities,
            'assignments': assignments,
            'prototype_embeddings': prototype_embeddings,
            'hard_assignments': hard_assignments,
        }

        # Optionally compute Euclidean distances
        if return_distances:
            # (B, T, 1, D) - (1, 1, P, D) -> (B, T, P, D) -> (B, T, P)
            distances = torch.norm(
                h.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0),
                dim=-1
            )
            outputs['distances'] = distances

        return outputs

    def get_prototype_similarities(self) -> torch.Tensor:
        """
        Compute pairwise similarities between all prototypes.

        Used to monitor prototype diversity during training.
        Low average similarity indicates diverse prototypes.

        Returns:
            (P, P) matrix of pairwise cosine similarities
        """
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim_matrix = torch.matmul(p_norm, p_norm.t())
        return sim_matrix

    def get_prototype_stats(self) -> Dict[str, float]:
        """
        Get statistics about prototype diversity.

        Returns:
            Dictionary with:
                - avg_similarity: Average pairwise similarity (target: < 0.3)
                - max_similarity: Maximum pairwise similarity
                - min_similarity: Minimum pairwise similarity
        """
        sim_matrix = self.get_prototype_similarities()

        # Exclude diagonal (self-similarity = 1)
        mask = ~torch.eye(self.num_prototypes, dtype=torch.bool, device=sim_matrix.device)
        off_diag = sim_matrix[mask]

        return {
            'avg_similarity': off_diag.mean().item(),
            'max_similarity': off_diag.max().item(),
            'min_similarity': off_diag.min().item(),
        }

    def compute_utilization(
        self,
        assignments: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute prototype utilization statistics.

        Args:
            assignments: Soft assignments (B, T, P) from forward pass

        Returns:
            Dictionary with:
                - effective_prototypes: exp(entropy) of usage distribution
                - max_utilization: Maximum usage of any prototype
                - min_utilization: Minimum usage of any prototype
                - unused_count: Number of prototypes with < 1% usage
        """
        # Average assignment across batch and time
        avg_assignment = assignments.mean(dim=(0, 1))  # (P,)

        # Entropy of usage distribution
        # Higher entropy = more even usage
        entropy = -(avg_assignment * torch.log(avg_assignment + 1e-10)).sum()
        effective = torch.exp(entropy).item()

        return {
            'effective_prototypes': effective,
            'max_utilization': avg_assignment.max().item(),
            'min_utilization': avg_assignment.min().item(),
            'unused_count': (avg_assignment < 0.01).sum().item(),
        }


class GumbelPrototypeLayer(PrototypeLayer):
    """
    Prototype Layer with Gumbel-Softmax for hard assignments during training.

    Uses Gumbel-Softmax trick to enable gradient flow through discrete
    prototype assignments. Useful for tasks requiring hard clustering.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_prototypes: int = 100,
        temperature: float = 0.1,
        gumbel_temperature: float = 1.0,
        hard: bool = False,
        **kwargs,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_prototypes=num_prototypes,
            temperature=temperature,
            **kwargs,
        )
        self.gumbel_temperature = gumbel_temperature
        self.hard = hard

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_distances: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward with Gumbel-Softmax assignments."""
        B, T, D = hidden_states.shape

        # Project and normalize
        h = self.projection(hidden_states)
        h = self.dropout(h)
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        # Compute similarities (logits for Gumbel)
        similarities = torch.matmul(h_norm, p_norm.t())
        logits = similarities / self.temperature

        # Gumbel-Softmax for differentiable discrete assignments
        if self.training:
            assignments = F.gumbel_softmax(
                logits,
                tau=self.gumbel_temperature,
                hard=self.hard,
                dim=-1,
            )
        else:
            # During inference, use regular softmax
            assignments = F.softmax(logits, dim=-1)

        prototype_embeddings = torch.matmul(assignments, self.prototypes)
        hard_assignments = similarities.argmax(dim=-1)

        outputs = {
            'similarities': similarities,
            'assignments': assignments,
            'prototype_embeddings': prototype_embeddings,
            'hard_assignments': hard_assignments,
        }

        if return_distances:
            distances = torch.norm(
                h.unsqueeze(2) - self.prototypes.unsqueeze(0).unsqueeze(0),
                dim=-1
            )
            outputs['distances'] = distances

        return outputs
