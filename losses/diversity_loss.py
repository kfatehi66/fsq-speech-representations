"""
Diversity Loss - Prevents Prototype and Representation Collapse

This loss is needed to prevent the model from collapsing to trivial solutions
(all prototypes becoming identical, or all frames being assigned to the same
prototype).

Design notes:
1. The diversity loss has to be computed and added to the total loss
2. Diversity metrics should be monitored during training
3. Regularisation should be strong early in training

Components:
1. EntropyLoss: Encourage even prototype usage (prevent dead prototypes)
2. OrthogonalityLoss: Encourage diverse prototype vectors
3. TemporalDiversityLoss: Encourage different frames to use different prototypes
"""

import math
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyLoss(nn.Module):
    """
    Entropy Loss to encourage even prototype utilization.

    Maximizes entropy of the average prototype assignment distribution.
    This prevents "dead prototypes" that are never used.

    L_entropy = max(0, ρ - H(p̄))

    Where:
    - p̄ = average prototype assignment across batch
    - H(p̄) = entropy of p̄
    - ρ = target entropy (typically 0.9 * log(num_prototypes))

    Args:
        num_prototypes: Number of prototypes
        target_entropy_ratio: Ratio of maximum entropy to target (default: 0.9)
    """

    def __init__(
        self,
        num_prototypes: int = 100,
        target_entropy_ratio: float = 0.9,
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        max_entropy = math.log(num_prototypes)
        self.target_entropy = target_entropy_ratio * max_entropy

    def forward(
        self,
        assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute entropy loss.

        Args:
            assignments: Soft prototype assignments (B, T, P)

        Returns:
            loss: Scalar entropy loss
            stats: Dictionary with entropy statistics
        """
        # Average assignment across batch and time -> (P,)
        avg_assignment = assignments.mean(dim=(0, 1))

        # Ensure valid probability distribution
        avg_assignment = avg_assignment + 1e-10
        avg_assignment = avg_assignment / avg_assignment.sum()

        # Compute entropy
        entropy = -(avg_assignment * torch.log(avg_assignment)).sum()

        # Hinge loss: penalize if entropy is below target
        loss = F.relu(self.target_entropy - entropy)

        # Compute effective number of prototypes
        effective_prototypes = torch.exp(entropy).item()

        stats = {
            'entropy': entropy.item(),
            'effective_prototypes': effective_prototypes,
            'utilization_ratio': effective_prototypes / self.num_prototypes,
            'max_assignment': avg_assignment.max().item(),
            'min_assignment': avg_assignment.min().item(),
        }

        return loss, stats


class OrthogonalityLoss(nn.Module):
    """
    Orthogonality Loss to encourage diverse prototype vectors.

    Minimizes pairwise cosine similarity between prototypes.
    This prevents prototypes from collapsing to similar vectors.

    L_orth = (1/P(P-1)) * Σ_{i≠j} (p̂_i · p̂_j)²

    Where:
    - p̂_i = normalized prototype i
    - The sum is over all pairs of distinct prototypes
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute orthogonality loss.

        Args:
            prototypes: Prototype vectors (P, D)

        Returns:
            loss: Scalar orthogonality loss
            stats: Dictionary with similarity statistics
        """
        P = prototypes.size(0)

        # Normalize prototypes
        p_norm = F.normalize(prototypes, dim=-1)

        # Compute pairwise similarities
        sim_matrix = torch.matmul(p_norm, p_norm.t())  # (P, P)

        # Mask diagonal (self-similarity)
        mask = ~torch.eye(P, dtype=torch.bool, device=prototypes.device)
        off_diag = sim_matrix[mask]

        # Loss: squared similarities (penalize high similarity)
        loss = (off_diag ** 2).mean()

        stats = {
            'avg_similarity': off_diag.abs().mean().item(),
            'max_similarity': off_diag.abs().max().item(),
            'min_similarity': off_diag.abs().min().item(),
        }

        return loss, stats


class TemporalDiversityLoss(nn.Module):
    """
    Temporal Diversity Loss to encourage different assignments over time.

    Prevents the model from assigning all frames to the same prototype,
    which would indicate representation collapse.

    L_temporal = max(0, ρ - Var(assignments))

    Where:
    - Var(assignments) = variance of hard assignments over time
    - ρ = minimum variance threshold
    """

    def __init__(self, min_variance: float = 0.1):
        super().__init__()
        self.min_variance = min_variance

    def forward(
        self,
        assignments: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute temporal diversity loss.

        Args:
            assignments: Soft prototype assignments (B, T, P)

        Returns:
            loss: Scalar temporal diversity loss
            stats: Dictionary with temporal statistics
        """
        # Hard assignments
        hard_assignments = assignments.argmax(dim=-1).float()  # (B, T)

        # Compute variance over time for each batch
        # Higher variance = more diverse assignments = good
        temporal_var = hard_assignments.var(dim=-1).mean()

        # Hinge loss: penalize if variance is too low
        loss = F.relu(self.min_variance - temporal_var)

        # Compute unique assignments per batch
        B, T = hard_assignments.shape
        unique_counts = []
        for b in range(B):
            unique_counts.append(hard_assignments[b].unique().numel())
        avg_unique = sum(unique_counts) / len(unique_counts)

        stats = {
            'temporal_variance': temporal_var.item(),
            'avg_unique_prototypes': avg_unique,
        }

        return loss, stats


class DiversityLoss(nn.Module):
    """
    Combined Diversity Loss.

    Combines entropy, orthogonality, and temporal diversity losses
    to prevent all forms of collapse.

    L_diversity = λ_ent * L_entropy + λ_orth * L_orthogonality + λ_temp * L_temporal

    Args:
        num_prototypes: Number of prototypes
        lambda_entropy: Weight for entropy loss (default: 1.0)
        lambda_orthogonality: Weight for orthogonality loss (default: 0.1)
        lambda_temporal: Weight for temporal diversity loss (default: 0.5)
        target_entropy_ratio: Target entropy as ratio of max (default: 0.9)
        min_temporal_variance: Minimum temporal variance (default: 0.1)
    """

    def __init__(
        self,
        num_prototypes: int = 100,
        lambda_entropy: float = 1.0,
        lambda_orthogonality: float = 0.1,
        lambda_temporal: float = 0.5,
        target_entropy_ratio: float = 0.9,
        min_temporal_variance: float = 0.1,
    ):
        super().__init__()

        self.lambda_entropy = lambda_entropy
        self.lambda_orthogonality = lambda_orthogonality
        self.lambda_temporal = lambda_temporal

        self.entropy_loss = EntropyLoss(
            num_prototypes=num_prototypes,
            target_entropy_ratio=target_entropy_ratio,
        )
        self.orthogonality_loss = OrthogonalityLoss()
        self.temporal_diversity_loss = TemporalDiversityLoss(
            min_variance=min_temporal_variance,
        )

    def forward(
        self,
        assignments: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined diversity loss.

        Args:
            assignments: Soft prototype assignments (B, T, P)
            prototypes: Prototype vectors (P, D)

        Returns:
            total_loss: Combined diversity loss
            losses: Dictionary with individual losses and stats
        """
        # Entropy loss
        entropy_loss, entropy_stats = self.entropy_loss(assignments)

        # Orthogonality loss
        orth_loss, orth_stats = self.orthogonality_loss(prototypes)

        # Temporal diversity loss
        temp_loss, temp_stats = self.temporal_diversity_loss(assignments)

        # Combined loss
        total_loss = (
            self.lambda_entropy * entropy_loss +
            self.lambda_orthogonality * orth_loss +
            self.lambda_temporal * temp_loss
        )

        losses = {
            'diversity_loss': total_loss,
            'entropy_loss': entropy_loss,
            'orthogonality_loss': orth_loss,
            'temporal_loss': temp_loss,
            # Stats for monitoring
            'entropy': torch.tensor(entropy_stats['entropy']),
            'effective_prototypes': torch.tensor(entropy_stats['effective_prototypes']),
            'avg_proto_similarity': torch.tensor(orth_stats['avg_similarity']),
            'temporal_variance': torch.tensor(temp_stats['temporal_variance']),
        }

        return total_loss, losses

    def check_collapse(
        self,
        assignments: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Dict[str, bool]:
        """
        Check for signs of collapse.

        Returns dictionary of warning flags.
        """
        _, entropy_stats = self.entropy_loss(assignments)
        _, orth_stats = self.orthogonality_loss(prototypes)
        _, temp_stats = self.temporal_diversity_loss(assignments)

        warnings = {
            'low_utilization': entropy_stats['utilization_ratio'] < 0.5,
            'high_proto_similarity': orth_stats['avg_similarity'] > 0.5,
            'low_temporal_diversity': temp_stats['temporal_variance'] < 0.05,
        }

        return warnings


class SinkhornKnopp(nn.Module):
    """
    Sinkhorn-Knopp algorithm for balanced prototype assignments.

    It counters collapse, as in SwAV, ProtoPool and other prototype methods.

    Forces each prototype to be used equally across the batch while
    maintaining row-stochastic property (each frame's assignments sum to 1).
    """

    def __init__(self, num_iterations: int = 3, epsilon: float = 0.1):
        super().__init__()
        self.num_iterations = num_iterations
        self.epsilon = epsilon

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Apply Sinkhorn normalization to assignment scores.

        Args:
            scores: (batch * seq_len, num_prototypes) similarity scores

        Returns:
            Balanced soft assignments (same shape), row-stochastic
        """
        N, K = scores.shape  # N = num samples, K = num prototypes

        with torch.no_grad():
            # Subtract max for numerical stability before exp
            scores_stable = scores - scores.max(dim=1, keepdim=True)[0]

            # Convert to probabilities with temperature
            Q = torch.exp(scores_stable / self.epsilon)

            # Add small epsilon for numerical stability
            Q = Q + 1e-10

            # Sinkhorn iterations
            # Goal: make columns sum to N/K (equal prototype usage)
            # while keeping rows summing to 1 (valid assignments)
            for _ in range(self.num_iterations):
                # Normalize columns: each prototype gets equal total assignment
                col_sum = Q.sum(dim=0, keepdim=True)
                Q = Q / col_sum * (N / K)

                # Normalize rows: each sample's assignments sum to 1
                row_sum = Q.sum(dim=1, keepdim=True)
                Q = Q / row_sum

        return Q
