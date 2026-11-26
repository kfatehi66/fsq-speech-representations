#!/usr/bin/env python3
"""
Temporal Losses for ProtoSpeech-TD (Temporally Diverse).

These losses prevent the "temporal smoothing" collapse where the model
learns to predict "same prototype as nearby frames" instead of
"prototype that represents this sound."

Three components:
1. TemporalTransitionLoss - Force prototype changes at boundaries
2. CrossTimeContrastiveLoss - Distinguish content from timestamp
3. PrototypeDiversityLoss - Use all prototypes (prevent A-B-A-B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalTransitionLoss(nn.Module):
    """
    Penalize excessive self-transitions (temporal smoothing).

    The model can "cheat" by outputting the same prototype for consecutive
    frames (a high self-transition rate). This loss forces transitions to occur
    at a rate similar to phoneme boundaries (~35%).

    Uses hinge loss: only penalize if transition rate < (target - margin).
    This allows naturally variable speech rates.
    """

    def __init__(
        self,
        target_transition_rate: float = 0.35,
        margin: float = 0.05,
        warmup_steps: int = 500
    ):
        super().__init__()
        self.target_rate = target_transition_rate
        self.margin = margin
        self.warmup_steps = warmup_steps
        self.register_buffer('step', torch.tensor(0, dtype=torch.long))

    def forward(self, assignments: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            assignments: (B, T, P) soft prototype assignments

        Returns:
            loss: scalar tensor
            transition_rate: float for logging
        """
        # Get hard assignments
        hard_assign = assignments.argmax(dim=-1)  # (B, T)
        B, T = hard_assign.shape

        if T < 2:
            return torch.tensor(0.0, device=assignments.device), 0.0

        # Compute transitions (where prototype changes)
        transitions = (hard_assign[:, 1:] != hard_assign[:, :-1]).float()  # (B, T-1)
        transition_rate = transitions.mean().item()

        # Hinge loss: penalize only if too smooth
        min_acceptable = self.target_rate - self.margin

        if transition_rate < min_acceptable:
            # Scale by how far below threshold
            deficit = min_acceptable - transition_rate
            loss = torch.tensor(deficit ** 2 * 10.0, device=assignments.device, requires_grad=True)
        else:
            loss = torch.tensor(0.0, device=assignments.device)

        # Warmup: gradually increase loss weight
        self.step += 1
        if self.step.item() < self.warmup_steps:
            warmup_scale = self.step.item() / self.warmup_steps
            loss = loss * warmup_scale

        return loss, transition_rate


class CrossTimeContrastiveLoss(nn.Module):
    """
    Contrastive loss with negatives from distant time steps.

    Standard contrastive learning uses random negatives, which allows
    the model to encode "timestamp" rather than "content."

    By forcing negatives to come from |t - t'| > min_gap, we force
    the model to encode content (which is what distinguishes frames
    that are far apart in time).

    Positive: (frame_t, assigned_prototype_t)
    Negative: (frame_t, assigned_prototype_{t'}) where |t-t'| > min_gap
    """

    def __init__(
        self,
        temperature: float = 0.1,
        min_time_gap: int = 20,
        num_negatives: int = 32
    ):
        super().__init__()
        self.temperature = temperature
        self.min_time_gap = min_time_gap
        self.num_negatives = num_negatives

    def forward(
        self,
        hidden_states: torch.Tensor,
        prototype_embeddings: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D) encoder outputs
            prototype_embeddings: (B, T, D) assigned prototype vectors
            prototypes: (P, D) all prototype vectors

        Returns:
            loss: scalar contrastive loss
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device

        if T < self.min_time_gap * 2:
            return torch.tensor(0.0, device=device)

        # Normalize for cosine similarity
        h_norm = F.normalize(hidden_states, dim=-1)  # (B, T, D)
        p_norm = F.normalize(prototype_embeddings, dim=-1)  # (B, T, D)

        # Sample anchor frames (not too close to edges)
        valid_range = T - 2 * self.min_time_gap
        if valid_range < 1:
            return torch.tensor(0.0, device=device)

        num_anchors = min(64, valid_range)
        anchor_offsets = torch.randperm(valid_range, device=device)[:num_anchors]
        anchor_idx = anchor_offsets + self.min_time_gap  # (num_anchors,)

        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            for anchor in anchor_idx:
                anchor = anchor.item()

                # Positive: the prototype assigned to this frame
                h_anchor = h_norm[b, anchor]  # (D,)
                p_positive = p_norm[b, anchor]  # (D,)

                # Positive similarity
                pos_sim = (h_anchor * p_positive).sum() / self.temperature

                # Negatives: prototypes from distant time steps
                neg_indices = []
                for t in range(T):
                    if abs(t - anchor) > self.min_time_gap:
                        neg_indices.append(t)

                if len(neg_indices) < self.num_negatives:
                    continue

                # Sample negatives
                neg_sample_idx = torch.tensor(
                    neg_indices, device=device
                )[torch.randperm(len(neg_indices), device=device)[:self.num_negatives]]

                neg_protos = p_norm[b, neg_sample_idx]  # (num_neg, D)
                neg_sims = (h_anchor.unsqueeze(0) * neg_protos).sum(dim=-1) / self.temperature  # (num_neg,)

                # InfoNCE loss
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # (1 + num_neg,)
                labels = torch.zeros(1, dtype=torch.long, device=device)  # Positive is index 0

                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                total_loss = total_loss + loss
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / count


class PrototypeDiversityLoss(nn.Module):
    """
    Entropy-based loss to ensure all prototypes are used.

    Without this, the TransitionLoss might cause the model to just
    oscillate between 2 prototypes (A-B-A-B pattern).

    This loss penalizes low entropy in prototype usage distribution.
    Target: use all K prototypes somewhat evenly.
    """

    def __init__(
        self,
        num_prototypes: int = 50,
        target_entropy_ratio: float = 0.7  # 70% of max entropy
    ):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.max_entropy = torch.log(torch.tensor(float(num_prototypes)))
        self.target_ratio = target_entropy_ratio

    def forward(self, assignments: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Args:
            assignments: (B, T, P) soft prototype assignments

        Returns:
            loss: scalar tensor
            entropy_ratio: float for logging (0-1, higher is more diverse)
        """
        device = assignments.device

        # Average assignment across batch and time
        avg_assign = assignments.mean(dim=(0, 1))  # (P,)

        # Add small epsilon for numerical stability
        avg_assign = avg_assign + 1e-10
        avg_assign = avg_assign / avg_assign.sum()  # Renormalize

        # Compute entropy
        entropy = -(avg_assign * avg_assign.log()).sum()
        max_entropy = self.max_entropy.to(device)
        entropy_ratio = (entropy / max_entropy).item()

        # Penalize if entropy is below target
        if entropy_ratio < self.target_ratio:
            deficit = self.target_ratio - entropy_ratio
            loss = torch.tensor(deficit ** 2 * 5.0, device=device, requires_grad=True)
        else:
            loss = torch.tensor(0.0, device=device)

        return loss, entropy_ratio


class TemporalDiversityLoss(nn.Module):
    """
    Combined temporal diversity loss for ProtoSpeech-TD.

    Wraps all three components with configurable weights.
    """

    def __init__(
        self,
        num_prototypes: int = 50,
        lambda_transition: float = 1.0,
        lambda_crosstime: float = 0.5,
        lambda_diversity: float = 0.1,
        target_transition_rate: float = 0.35,
        min_time_gap: int = 20
    ):
        super().__init__()

        self.lambda_transition = lambda_transition
        self.lambda_crosstime = lambda_crosstime
        self.lambda_diversity = lambda_diversity

        self.transition_loss = TemporalTransitionLoss(
            target_transition_rate=target_transition_rate
        )
        self.crosstime_loss = CrossTimeContrastiveLoss(
            min_time_gap=min_time_gap
        )
        self.diversity_loss = PrototypeDiversityLoss(
            num_prototypes=num_prototypes
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prototype_embeddings: torch.Tensor,
        assignments: torch.Tensor,
        prototypes: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined temporal diversity loss.

        Returns:
            total_loss: weighted sum of all components
            metrics: dict with individual losses and rates for logging
        """
        # Transition loss
        trans_loss, trans_rate = self.transition_loss(assignments)

        # Cross-time contrastive loss
        crosstime_loss = self.crosstime_loss(
            hidden_states, prototype_embeddings, prototypes
        )

        # Diversity loss
        div_loss, entropy_ratio = self.diversity_loss(assignments)

        # Weighted sum
        total_loss = (
            self.lambda_transition * trans_loss +
            self.lambda_crosstime * crosstime_loss +
            self.lambda_diversity * div_loss
        )

        metrics = {
            'transition_loss': trans_loss.item() if torch.is_tensor(trans_loss) else trans_loss,
            'crosstime_loss': crosstime_loss.item() if torch.is_tensor(crosstime_loss) else crosstime_loss,
            'diversity_loss': div_loss.item() if torch.is_tensor(div_loss) else div_loss,
            'transition_rate': trans_rate,
            'entropy_ratio': entropy_ratio,
        }

        return total_loss, metrics
