"""
Contrastive Loss for view invariance.

Standard InfoNCE contrastive loss to encourage representations
to be invariant to data augmentation.
"""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    InfoNCE Contrastive Loss.

    L = -log[exp(sim(z_i, z_i+) / τ) / Σ_k exp(sim(z_i, z_k) / τ)]

    Positive pairs: Same frame from different augmented views
    Negative pairs: Different frames

    Args:
        temperature: Temperature for softmax (default: 0.1)
        use_prototype_embeddings: If True, use prototype-weighted embeddings
                                  instead of raw hidden states
    """

    def __init__(
        self,
        temperature: float = 0.1,
        use_prototype_embeddings: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_prototype_embeddings = use_prototype_embeddings

    def forward(
        self,
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute contrastive loss.

        Args:
            embeddings_1: Embeddings from view 1 (B, T, D)
            embeddings_2: Embeddings from view 2 (B, T, D)

        Returns:
            loss: Scalar contrastive loss
            stats: Dictionary with statistics
        """
        B, T, D = embeddings_1.shape

        # Flatten to (B*T, D)
        z1 = embeddings_1.reshape(-1, D)
        z2 = embeddings_2.reshape(-1, D)

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Positive similarities: diagonal of z1 @ z2.T
        # (B*T,) - each position compared to its augmented version
        pos_sim = (z1 * z2).sum(dim=-1) / self.temperature

        # All pairwise similarities for negatives
        # (B*T, B*T)
        all_sim = torch.matmul(z1, z2.t()) / self.temperature

        # InfoNCE: denominator is sum over all pairs
        # log_sum_exp over all similarities
        log_denominator = torch.logsumexp(all_sim, dim=-1)

        # Loss = -positive + log(sum(exp(all)))
        loss = (-pos_sim + log_denominator).mean()

        # Compute accuracy (is positive pair most similar?)
        pred_positive = all_sim.argmax(dim=-1)
        targets = torch.arange(z1.size(0), device=z1.device)
        accuracy = (pred_positive == targets).float().mean().item()

        # Average positive and negative similarities
        mask = torch.eye(z1.size(0), dtype=torch.bool, device=z1.device)
        pos_mean = (all_sim * mask.float()).sum() / mask.sum()
        neg_mean = (all_sim * (~mask).float()).sum() / (~mask).sum()

        stats = {
            'contrastive_loss': loss.item(),
            'contrastive_accuracy': accuracy,
            'avg_pos_similarity': pos_mean.item() * self.temperature,
            'avg_neg_similarity': neg_mean.item() * self.temperature,
        }

        return loss, stats


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.

    Variant of contrastive loss used in SimCLR.
    Similar to InfoNCE but with symmetric loss computation.

    Args:
        temperature: Temperature for softmax (default: 0.5)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_1: torch.Tensor,
        embeddings_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute NT-Xent loss.

        Args:
            embeddings_1: Embeddings from view 1 (B, T, D) or (N, D)
            embeddings_2: Embeddings from view 2 (B, T, D) or (N, D)

        Returns:
            loss: Scalar contrastive loss
            stats: Dictionary with statistics
        """
        # Handle both (B, T, D) and (N, D) inputs
        if embeddings_1.dim() == 3:
            z1 = embeddings_1.reshape(-1, embeddings_1.size(-1))
            z2 = embeddings_2.reshape(-1, embeddings_2.size(-1))
        else:
            z1, z2 = embeddings_1, embeddings_2

        N = z1.size(0)
        D = z1.size(1)

        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Concatenate: [z1; z2] of shape (2N, D)
        representations = torch.cat([z1, z2], dim=0)

        # Similarity matrix (2N, 2N)
        similarity = torch.matmul(representations, representations.t()) / self.temperature

        # Create mask for positive pairs
        # Positive pairs: (i, i+N) and (i+N, i)
        labels = torch.arange(N, device=z1.device)
        labels = torch.cat([labels + N, labels], dim=0)  # (2N,)

        # Mask out self-similarity
        mask = torch.eye(2 * N, dtype=torch.bool, device=z1.device)
        similarity = similarity.masked_fill(mask, float('-inf'))

        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)

        # Compute accuracy
        pred = similarity.argmax(dim=-1)
        accuracy = (pred == labels).float().mean().item()

        stats = {
            'ntxent_loss': loss.item(),
            'contrastive_accuracy': accuracy,
        }

        return loss, stats
