"""
Prototype Losses - Cluster and Separation

These losses encourage frames to cluster around their nearest prototype
while staying away from other prototypes.

Inspired by ProtoPNet (Chen et al., NeurIPS 2019):
- ClusterLoss: Pull frames towards their nearest prototype
- SeparationLoss: Push frames away from non-matching prototypes
"""

from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClusterLoss(nn.Module):
    """
    Cluster Loss - Pull frames towards their assigned prototypes.

    L_cluster = (1/BT) * Σ_{b,i} min_j ||h_{bi} - p_j||²

    Each frame is pulled towards its nearest prototype.

    Args:
        use_cosine: Use cosine distance instead of Euclidean
    """

    def __init__(self, use_cosine: bool = False):
        super().__init__()
        self.use_cosine = use_cosine

    def forward(
        self,
        hidden_states: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cluster loss.

        Args:
            hidden_states: Frame embeddings (B, T, D)
            prototypes: Prototype vectors (P, D)
            assignments: Optional soft assignments (B, T, P) - if provided,
                        use weighted distance instead of min

        Returns:
            loss: Scalar cluster loss
            stats: Dictionary with statistics
        """
        B, T, D = hidden_states.shape
        P = prototypes.size(0)

        if self.use_cosine:
            # Cosine distance: 1 - cosine_similarity
            h_norm = F.normalize(hidden_states, dim=-1)  # (B, T, D)
            p_norm = F.normalize(prototypes, dim=-1)  # (P, D)
            similarities = torch.matmul(h_norm, p_norm.t())  # (B, T, P)
            distances = 1 - similarities
        else:
            # Squared Euclidean distance
            # (B, T, 1, D) - (1, 1, P, D) -> (B, T, P)
            diff = hidden_states.unsqueeze(2) - prototypes.unsqueeze(0).unsqueeze(0)
            distances = (diff ** 2).sum(dim=-1)

        if assignments is not None:
            # Weighted distance using soft assignments
            loss = (assignments * distances).sum(dim=-1).mean()
        else:
            # Minimum distance to any prototype
            min_distances, _ = distances.min(dim=-1)  # (B, T)
            loss = min_distances.mean()

        stats = {
            'avg_min_distance': distances.min(dim=-1)[0].mean().item(),
            'avg_distance': distances.mean().item(),
        }

        return loss, stats


class SeparationLoss(nn.Module):
    """
    Separation Loss - Push frames away from non-matching prototypes.

    L_sep = (1/BT) * Σ_{b,i} max(0, m - d(h_{bi}, p_{nearest_wrong}))

    Each frame is pushed away from the nearest wrong prototype
    (prototypes other than the one it's assigned to).

    Args:
        margin: Minimum distance to maintain from wrong prototypes
        use_cosine: Use cosine distance instead of Euclidean
    """

    def __init__(self, margin: float = 0.5, use_cosine: bool = False):
        super().__init__()
        self.margin = margin
        self.use_cosine = use_cosine

    def forward(
        self,
        hidden_states: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute separation loss.

        Args:
            hidden_states: Frame embeddings (B, T, D)
            prototypes: Prototype vectors (P, D)
            assignments: Soft assignments (B, T, P)

        Returns:
            loss: Scalar separation loss
            stats: Dictionary with statistics
        """
        B, T, D = hidden_states.shape
        P = prototypes.size(0)

        if self.use_cosine:
            h_norm = F.normalize(hidden_states, dim=-1)
            p_norm = F.normalize(prototypes, dim=-1)
            similarities = torch.matmul(h_norm, p_norm.t())
            distances = 1 - similarities
        else:
            diff = hidden_states.unsqueeze(2) - prototypes.unsqueeze(0).unsqueeze(0)
            distances = (diff ** 2).sum(dim=-1)

        # Get assigned prototype for each frame
        assigned_proto = assignments.argmax(dim=-1)  # (B, T)

        # Create mask for non-assigned prototypes
        # (B, T, P) mask where True = not the assigned prototype
        proto_indices = torch.arange(P, device=hidden_states.device)
        non_assigned_mask = proto_indices.unsqueeze(0).unsqueeze(0) != assigned_proto.unsqueeze(-1)

        # Set assigned prototype distance to large value so it's not selected
        masked_distances = distances.clone()
        masked_distances[~non_assigned_mask] = float('inf')

        # Find nearest wrong prototype
        nearest_wrong_dist, _ = masked_distances.min(dim=-1)  # (B, T)

        # Hinge loss: penalize if nearest wrong prototype is too close
        loss = F.relu(self.margin - nearest_wrong_dist).mean()

        stats = {
            'avg_nearest_wrong_dist': nearest_wrong_dist[nearest_wrong_dist < float('inf')].mean().item(),
            'fraction_within_margin': (nearest_wrong_dist < self.margin).float().mean().item(),
        }

        return loss, stats


class PrototypeLoss(nn.Module):
    """
    Combined Prototype Loss.

    L_prototype = λ_cluster * L_cluster + λ_sep * L_separation

    Args:
        lambda_cluster: Weight for cluster loss (default: 1.0)
        lambda_separation: Weight for separation loss (default: 0.5)
        separation_margin: Margin for separation loss (default: 0.5)
        use_cosine: Use cosine distance (default: False)
    """

    def __init__(
        self,
        lambda_cluster: float = 1.0,
        lambda_separation: float = 0.5,
        separation_margin: float = 0.5,
        use_cosine: bool = False,
    ):
        super().__init__()

        self.lambda_cluster = lambda_cluster
        self.lambda_separation = lambda_separation

        self.cluster_loss = ClusterLoss(use_cosine=use_cosine)
        self.separation_loss = SeparationLoss(
            margin=separation_margin,
            use_cosine=use_cosine,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prototypes: torch.Tensor,
        assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined prototype loss.

        Args:
            hidden_states: Frame embeddings (B, T, D)
            prototypes: Prototype vectors (P, D)
            assignments: Soft assignments (B, T, P)

        Returns:
            total_loss: Combined prototype loss
            losses: Dictionary with individual losses
        """
        # Cluster loss
        cluster_loss, cluster_stats = self.cluster_loss(
            hidden_states, prototypes, assignments
        )

        # Separation loss
        sep_loss, sep_stats = self.separation_loss(
            hidden_states, prototypes, assignments
        )

        # Combined loss
        total_loss = (
            self.lambda_cluster * cluster_loss +
            self.lambda_separation * sep_loss
        )

        losses = {
            'prototype_loss': total_loss,
            'cluster_loss': cluster_loss,
            'separation_loss': sep_loss,
            'avg_min_distance': torch.tensor(cluster_stats['avg_min_distance']),
            'fraction_within_margin': torch.tensor(sep_stats['fraction_within_margin']),
        }

        return total_loss, losses
