"""
wav2vec 2.0 contrastive objective over quantised targets.

Baevski et al. 2020, "wav2vec 2.0: A Framework for Self-Supervised Learning of
Speech Representations". Section 3.2:

    L_m = -log  exp(sim(c_t, q_t) / kappa)
                --------------------------------
                sum_{q~ in Q_t} exp(sim(c_t, q~) / kappa)

where c_t is the Transformer context at a masked timestep t, q_t is the
quantised target at that timestep, and Q_t is {q_t} together with K distractors
drawn uniformly from the other masked timesteps of the same utterance.
sim is cosine similarity; kappa = 0.1.

Difference from losses/contrastive_loss.py
------------------------------------------
That module computes `z1 @ z2.T` over all B*T frames of two augmented views:
cross-view, whole-batch negatives. wav2vec 2.0 needs within-utterance negatives
restricted to masked positions, and a single view. The negative distribution is
part of the objective here, not a configuration detail: sampling negatives from
other utterances makes the task solvable by speaker identity alone.

Diversity loss
--------------
`CodebookDiversityLoss` implements wav2vec 2.0's L_d:

    L_d = (G*V - sum_g exp(H(p_g))) / (G*V)

the normalised gap between maximum and actual codebook perplexity, averaged
over G groups of V entries. It is wired to the learned-codebook branches only:

  * gumbel_pq: the native case. Gradients flow through the Gumbel-softmax
    probabilities exactly as in the paper.
  * rvq4l: adapted. Hard nearest-neighbour assignment has no gradient, so
    the probabilities are taken as softmax(-distances / tau) rather than the
    one-hot argmin. This keeps L_d a real regulariser instead of a monitor. The
    assignment used for the targets is still hard argmin; only L_d sees the
    soft distribution.
  * fsq16d: not used, by design. FSQ has no codebook to collapse: its levels
    are fixed by construction, so there is no "dead code" failure mode for L_d
    to prevent and no learned parameter for it to act on. A diversity term on
    the FSQ branch would be a loss with no referent. This asymmetry is part of
    the comparison.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_negative_indices(mask: torch.Tensor, num_negatives: int,
                            generator: torch.Generator = None):
    """Draw K distractor positions per masked frame, from the same utterance.

    Args:
        mask: (B, T) bool, True at masked timesteps.
        num_negatives: K.
        generator: optional torch.Generator for reproducibility.

    Returns:
        neg_flat: (N, K) long. Indices into the flattened masked-frame array,
            i.e. into `x[mask]` of shape (N, D), where N = mask.sum(). Every
            entry indexes a masked frame of the same utterance as its row, and
            never the row's own frame.
        usable: (N,) bool. False for rows whose utterance has fewer than 2
            masked frames: there is no valid distractor for those, and the
            caller must drop them from the loss.

    The self-exclusion is exact rather than rejection-sampled: draw
    u ~ Uniform{0..n-2} over the utterance's other frames, then shift by
    `u >= own_rank` to skip over the row's own position. This is fairseq's
    method. With rejection sampling a frame can be its own
    distractor, which puts the positive in the denominator twice and biases the
    loss downward.
    """
    B, T = mask.shape
    device = mask.device

    counts = mask.sum(dim=1)                                    # (B,)
    # Exclusive prefix sum: where utterance b's frames start in the flat array.
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])

    bt = mask.nonzero(as_tuple=False)                            # (N, 2)
    b_of = bt[:, 0]
    n_frames = bt.shape[0]

    if n_frames == 0:
        return (torch.zeros(0, num_negatives, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.bool, device=device))

    # Rank of each masked frame within its own utterance (0-based).
    own_rank = torch.arange(n_frames, device=device) - offsets[b_of]
    n_in_utt = counts[b_of]                                      # (N,)
    usable = n_in_utt >= 2

    # Uniform over the n-1 other frames. Clamp so utterances with n<2 produce a
    # well-formed (in-bounds) index; those rows are excluded via `usable`.
    n_others = (n_in_utt - 1).clamp(min=1).unsqueeze(1)          # (N, 1)
    u = torch.rand(n_frames, num_negatives, device=device, generator=generator)
    u = (u * n_others).long().clamp(max=n_others - 1)            # {0..n-2}

    # Skip over the row's own rank.
    neg_rank = u + (u >= own_rank.unsqueeze(1)).long()
    neg_flat = offsets[b_of].unsqueeze(1) + neg_rank
    return neg_flat, usable


class W2V2ContrastiveLoss(nn.Module):
    """InfoNCE over quantised targets with within-utterance masked negatives.

    Args:
        num_negatives: K distractors per masked frame (paper: 100).
        temperature: kappa (paper: 0.1).
    """

    def __init__(self, num_negatives: int = 100, temperature: float = 0.1):
        super().__init__()
        self.num_negatives = num_negatives
        self.temperature = temperature

    def forward(self, context: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, generator: torch.Generator = None):
        """
        Args:
            context: (B, T, D) Transformer output c.
            target:  (B, T, D) quantised targets q.
            mask:    (B, T) bool, True at masked timesteps.

        Returns:
            loss: scalar. Zero (with requires_grad preserved) if no masked frame
                has a valid distractor.
            stats: dict of floats for logging.
        """
        c = context[mask]                                        # (N, D)
        q = target[mask]                                         # (N, D)

        neg_flat, usable = sample_negative_indices(
            mask, self.num_negatives, generator=generator)

        if usable.sum() == 0:
            zero = context.sum() * 0.0
            return zero, {'contrastive_loss': 0.0, 'accuracy': 0.0,
                          'n_masked': int(mask.sum()), 'n_usable': 0}

        c, q, neg_flat = c[usable], q[usable], neg_flat[usable]
        negatives = q[neg_flat]                                  # (N, K, D)

        c_n = F.normalize(c, dim=-1)
        q_n = F.normalize(q, dim=-1)
        neg_n = F.normalize(negatives, dim=-1)

        pos_logits = (c_n * q_n).sum(-1, keepdim=True)           # (N, 1)
        neg_logits = torch.bmm(neg_n, c_n.unsqueeze(-1)).squeeze(-1)  # (N, K)

        # A distractor can coincide with the positive when the quantiser maps
        # two masked frames to the same code, which is common early in
        # training, and increasingly common as a codebook collapses. Leaving
        # them in makes the task unsolvable in a way that looks like a bad
        # objective rather than a bad codebook, so they are masked out of the
        # denominator (as in fairseq).
        duplicate = (negatives == q.unsqueeze(1)).all(dim=-1)    # (N, K)
        neg_logits = neg_logits.masked_fill(duplicate, float('-inf'))

        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            acc = (logits.argmax(dim=1) == 0).float().mean().item()
            dup_rate = duplicate.float().mean().item()

        return loss, {
            'contrastive_loss': loss.item(),
            'accuracy': acc,
            'n_masked': int(mask.sum()),
            'n_usable': int(usable.sum()),
            'duplicate_negative_rate': dup_rate,
            'pos_cos_sim': pos_logits.mean().item() * 1.0,
        }


class CodebookDiversityLoss(nn.Module):
    """wav2vec 2.0 L_d: normalised codebook-perplexity gap.

        L_d = (G*V - sum_g exp(H(p_g))) / (G*V)

    p_g is the batch-averaged assignment distribution of group g. The loss is 0
    when every group uses all V entries uniformly and approaches 1 as groups
    collapse onto a single entry.

    Applies to learned codebooks only. The module docstring explains why the
    FSQ branch has none.
    """

    def __init__(self, num_groups: int, num_entries: int):
        super().__init__()
        self.num_groups = num_groups
        self.num_entries = num_entries

    def forward(self, probs: torch.Tensor):
        """
        Args:
            probs: (N, G, V) per-frame assignment distributions. Soft, so that
                the gradient reaches the encoder and the codebook. For rvq4l
                these are softmax(-distances / tau); for gumbel_pq they are the
                Gumbel-softmax probabilities.

        Returns:
            loss: scalar in [0, 1).
            stats: dict with per-group perplexity.
        """
        p = probs.mean(dim=0)                                    # (G, V)
        p = p / p.sum(dim=-1, keepdim=True).clamp(min=1e-10)
        h = -(p * torch.log(p.clamp(min=1e-10))).sum(dim=-1)     # (G,) nats
        perplexity = torch.exp(h)                                # (G,)

        gv = self.num_groups * self.num_entries
        loss = (gv - perplexity.sum()) / gv

        return loss, {
            'diversity_loss': loss.item(),
            'perplexity_per_group': [float(x) for x in perplexity.detach().cpu()],
            'mean_perplexity': float(perplexity.mean()),
            'max_perplexity': float(self.num_entries),
        }


def compute_span_mask(shape, mask_prob: float = 0.065, mask_length: int = 10,
                      device=None, generator: torch.Generator = None):
    """wav2vec 2.0 / HuBERT span masking.

    Defaults mask_prob 0.065, mask_length 10. Two properties are explicit:

      * `generator` is threaded through, so a seeded run masks reproducibly.
      * spans are guaranteed non-empty for T <= mask_length (otherwise a short
        utterance would get an all-False mask and contribute nothing to the
        contrastive loss).

    Spans may overlap, as in fairseq, so the realised masked fraction is
    slightly below num_spans*mask_length/T. With the defaults, the spans cover
    at most 6.5% of the frames of an utterance longer than about 3 s;
    wav2vec 2.0 uses 0.065 as the proportion of frames sampled as span starts,
    which masks about 49% of the frames (Baevski et al., 2020).

    Args:
        shape: (B, T).
        mask_prob: probability mass targeted; num_spans = mask_prob * T / L.
        mask_length: span length L.

    Returns:
        (B, T) bool mask.
    """
    B, T = shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    num_spans = max(1, int(mask_prob * T / mask_length))

    for b in range(B):
        if T > mask_length:
            starts = torch.randint(0, T - mask_length, (num_spans,),
                                   device=device, generator=generator)
            for s in starts:
                mask[b, s:min(int(s) + mask_length, T)] = True
        else:
            # Utterance shorter than one span: mask a single frame so the
            # utterance still contributes rather than silently dropping out.
            mask[b, 0] = True

    return mask
