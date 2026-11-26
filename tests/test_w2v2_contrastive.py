#!/usr/bin/env python3
"""
Tests for the wav2vec 2.0 contrastive objective and SSL encoder.

Obligations:

  1. NEGATIVE SAMPLING: distractors come from other masked timesteps of the
     same utterance, and never include the positive's own position.
  2. LOSS CORRECTNESS: the loss goes to ~0 when the context equals the target
     and the distractors are orthogonal to it; it sits at chance when the
     context carries no information.
  3. MASK STATISTICS:  the realised masking matches the configured rate and
     span length.
  4. POSITIONAL CONV:  with the relative positional convolution, two masked
     positions in the same utterance produce different context vectors;
     without it they are identical. This is the degeneracy argument of
     models/ssl_encoder.py, checked in code: an encoder
     with no positional information is permutation-equivariant, so every masked
     position (all carrying the same mask embedding) yields the same context,
     and the contrastive task becomes unsolvable in principle.

Run either way:
    python tests/test_w2v2_contrastive.py
    pytest -q tests/test_w2v2_contrastive.py
"""
import math
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from losses.w2v2_contrastive import (              # noqa: E402
    W2V2ContrastiveLoss, CodebookDiversityLoss,
    sample_negative_indices, compute_span_mask,
)
from models.ssl_encoder import SSLEncoder, RelativePositionalConv  # noqa: E402


# ----------------------------------------------------------------------------
# 1. NEGATIVE SAMPLING
# ----------------------------------------------------------------------------
def test_negatives_exclude_the_positive():
    """No sampled distractor may be the frame's own position."""
    torch.manual_seed(0)
    B, T, K = 4, 200, 50
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, ::3] = True                       # ~67 masked frames per utterance

    neg, usable = sample_negative_indices(mask, K)
    own = torch.arange(neg.shape[0]).unsqueeze(1)

    assert usable.all(), "every utterance here has >= 2 masked frames"
    assert not (neg == own).any(), (
        f"{int((neg == own).sum())} distractors were the frame's own position")


def test_negatives_stay_within_the_same_utterance():
    """Distractors must not leak across utterance boundaries.

    Cross-utterance negatives would let the model solve the task by speaker or
    channel identity instead of by phonetic content, which is exactly the
    failure mode wav2vec 2.0's within-utterance sampling avoids.
    """
    torch.manual_seed(1)
    B, T, K = 5, 120, 30
    mask = torch.zeros(B, T, dtype=torch.bool)
    # Deliberately uneven counts, so an off-by-one in the offsets shows up.
    for b in range(B):
        mask[b, : (b + 2) * 7] = True

    neg, usable = sample_negative_indices(mask, K)
    b_of = mask.nonzero()[:, 0]                       # utterance of each frame
    neg_b = b_of[neg]                                 # utterance of each distractor

    assert usable.all()
    assert torch.equal(neg_b, b_of.unsqueeze(1).expand_as(neg_b)), (
        "a distractor was drawn from a different utterance")


def test_negatives_only_from_masked_positions():
    """Every distractor index must map to a masked timestep."""
    torch.manual_seed(2)
    B, T, K = 3, 90, 20
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, 10:30] = True
    mask[:, 60:70] = True

    neg, _ = sample_negative_indices(mask, K)
    # Flat indices address x[mask], so any in-range index is masked by
    # construction; assert the range explicitly.
    n = int(mask.sum())
    assert int(neg.min()) >= 0 and int(neg.max()) < n, (
        f"distractor index out of range [0, {n})")


def test_single_masked_frame_utterance_is_marked_unusable():
    """An utterance with one masked frame has no valid distractor."""
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[0, 5] = True                    # 1 masked frame  -> unusable
    mask[1, 10:20] = True                # 10 masked frames -> usable

    neg, usable = sample_negative_indices(mask, 8)
    assert usable[0].item() is False
    assert usable[1:].all()
    # And in-range, so gathering never faults even on the excluded row.
    assert int(neg.min()) >= 0 and int(neg.max()) < int(mask.sum())


def test_negatives_cover_the_available_positions():
    """Sampling is roughly uniform, not stuck on one distractor."""
    torch.manual_seed(3)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    mask[0, :40] = True
    neg, _ = sample_negative_indices(mask, 100)
    # 40 frames x 100 draws: essentially all 40 positions should be hit.
    assert neg.unique().numel() >= 38, (
        f"only {neg.unique().numel()}/40 positions were ever sampled")


# ----------------------------------------------------------------------------
# 2. LOSS CORRECTNESS
# ----------------------------------------------------------------------------
def test_loss_goes_to_zero_for_perfect_match_with_orthogonal_distractors():
    """context == target, distractors orthogonal  =>  L_m -> 0."""
    B, T, D, K = 1, 64, 32, 16
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, :D] = True                    # D masked frames, one per basis vector

    # Each masked frame gets a distinct one-hot target: mutually orthogonal, so
    # every distractor has cosine similarity exactly 0 with the positive.
    target = torch.zeros(B, T, D)
    target[0, :D] = torch.eye(D)
    context = target.clone()              # perfect prediction

    loss, stats = W2V2ContrastiveLoss(num_negatives=K, temperature=0.1)(
        context, target, mask)

    # Closed form: logits are 1/kappa for the positive and 0 for K orthogonal
    # distractors, so L = log(1 + K*exp(-1/kappa)).
    expected = math.log(1 + K * math.exp(-1 / 0.1))
    assert abs(loss.item() - expected) < 1e-4, (
        f"loss {loss.item():.6f}, expected {expected:.6f}")
    assert loss.item() < 1e-3, f"loss did not approach 0: {loss.item()}"
    assert stats['accuracy'] == 1.0


def test_loss_is_at_chance_for_uninformative_context():
    """A context independent of the target sits at ~log(K+1)."""
    torch.manual_seed(7)
    B, T, D, K = 2, 128, 1024, 20
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, ::2] = True

    target = torch.randn(B, T, D)
    context = torch.randn(B, T, D)        # unrelated

    loss, stats = W2V2ContrastiveLoss(num_negatives=K, temperature=0.1)(
        context, target, mask)

    # D must be large for "chance" to mean log(K+1). The cosine similarity of
    # two iid gaussian vectors has variance 1/D, so the logits have variance
    # 1/(D*kappa^2) and Jensen puts E[loss] above log(K+1) by about half that:
    # ~0.78 nats at D=64, ~0.02 at D=1024. Measured worst deviation over 10
    # seeds at D=1024 is 0.117, so 0.25 is a real bound, not a rubber stamp.
    chance = math.log(K + 1)
    assert abs(loss.item() - chance) < 0.25, (
        f"uninformative context gave {loss.item():.3f}, chance is {chance:.3f}")
    assert stats['accuracy'] < 0.25


def test_loss_decreases_as_context_approaches_target():
    """The objective must actually be monotone in prediction quality."""
    torch.manual_seed(11)
    B, T, D, K = 2, 96, 48, 25
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[:, ::2] = True
    target = torch.randn(B, T, D)
    noise = torch.randn(B, T, D)

    crit = W2V2ContrastiveLoss(num_negatives=K, temperature=0.1)
    losses = []
    for alpha in (1.0, 0.5, 0.1, 0.0):     # noise weight: 1.0 = pure noise
        context = (1 - alpha) * target + alpha * noise
        losses.append(crit(context, target, mask)[0].item())

    assert losses == sorted(losses, reverse=True), (
        f"loss not monotonically decreasing with prediction quality: {losses}")


def test_loss_is_differentiable_wrt_context():
    torch.manual_seed(13)
    B, T, D = 2, 64, 32
    mask = torch.zeros(B, T, dtype=torch.bool); mask[:, ::2] = True
    context = torch.randn(B, T, D, requires_grad=True)
    target = torch.randn(B, T, D)
    loss, _ = W2V2ContrastiveLoss(num_negatives=10)(context, target, mask)
    loss.backward()
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert float(context.grad.abs().sum()) > 0


def test_diversity_loss_bounds():
    """L_d = 0 for a uniform codebook, -> 1 as it collapses."""
    G, V, N = 2, 320, 512
    crit = CodebookDiversityLoss(num_groups=G, num_entries=V)

    uniform = torch.full((N, G, V), 1.0 / V)
    loss_u, stats_u = crit(uniform)
    assert abs(loss_u.item()) < 1e-4, f"uniform L_d should be 0, got {loss_u.item()}"
    assert abs(stats_u['mean_perplexity'] - V) < 1e-1

    collapsed = torch.zeros(N, G, V); collapsed[:, :, 0] = 1.0
    loss_c, stats_c = crit(collapsed)
    assert loss_c.item() > 0.99, f"collapsed L_d should be ~1, got {loss_c.item()}"
    assert stats_c['mean_perplexity'] < 1.01


# ----------------------------------------------------------------------------
# 3. MASK STATISTICS
# ----------------------------------------------------------------------------
def test_mask_statistics_match_configuration():
    """Span count and span length must match mask_prob / mask_length."""
    torch.manual_seed(17)
    B, T = 8, 1000
    mask_prob, mask_length = 0.065, 10
    mask = compute_span_mask((B, T), mask_prob, mask_length)

    expected_spans = max(1, int(mask_prob * T / mask_length))     # 6
    max_frac = expected_spans * mask_length / T

    frac = mask.float().mean().item()
    # Spans may overlap (as in fairseq), so the realised
    # fraction is at most spans*L/T and in practice just below it.
    assert frac <= max_frac + 1e-6, f"masked {frac:.4f} > ceiling {max_frac:.4f}"
    assert frac > 0.7 * max_frac, (
        f"masked {frac:.4f}, far below the {max_frac:.4f} ceiling: spans are "
        f"overlapping much more than chance would give")

    # Every maximal run of True must be a multiple/merge of mask_length spans,
    # so no run may be shorter than mask_length.
    for b in range(B):
        row = mask[b].tolist()
        run = 0
        for i, v in enumerate(row):
            if v:
                run += 1
            else:
                if run:
                    assert run >= mask_length, (
                        f"utterance {b}: found a masked run of {run} < "
                        f"mask_length {mask_length}")
                run = 0


def test_mask_is_reproducible_under_a_seeded_generator():
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    m1 = compute_span_mask((4, 500), 0.065, 10, generator=g1)
    m2 = compute_span_mask((4, 500), 0.065, 10, generator=g2)
    assert torch.equal(m1, m2), "seeded masking is not reproducible"


def test_short_utterance_still_contributes():
    """T <= mask_length must not produce an all-False mask."""
    mask = compute_span_mask((3, 8), 0.065, 10)
    assert mask.any(dim=1).all(), "a short utterance was left entirely unmasked"


# ----------------------------------------------------------------------------
# 4. POSITIONAL CONV: the degeneracy argument
# ----------------------------------------------------------------------------
def _two_masked_contexts(use_positional_conv, seed=0):
    """Context vectors at two masked positions of one utterance."""
    torch.manual_seed(seed)
    model = SSLEncoder(quantiser='fsq16d',
                       use_positional_conv=use_positional_conv,
                       num_transformer_layers=2).eval()

    B, T, D = 1, 60, model.hidden_dim
    z = torch.randn(B, T, D)
    mask = torch.zeros(B, T, dtype=torch.bool)
    i, j = 10, 40
    mask[0, i] = True
    mask[0, j] = True

    with torch.no_grad():
        c = model.context(model.apply_mask(z, mask))
    return c[0, i], c[0, j]


def test_without_positional_conv_masked_contexts_are_identical():
    """No positional information => permutation-equivariant => degenerate.

    Both masked positions receive the same mask embedding, and a Transformer
    without positional information produces an output that depends only on
    (own input, multiset of all inputs). So the two contexts must coincide,
    and the contrastive objective, which asks the model to tell them apart,
    cannot be solved by any parameter setting.
    """
    ci, cj = _two_masked_contexts(use_positional_conv=False)
    assert torch.allclose(ci, cj, atol=1e-5), (
        f"expected identical contexts without positional information; "
        f"max |difference| = {float((ci - cj).abs().max()):.3e}")


def test_with_positional_conv_masked_contexts_differ():
    """The relative positional conv breaks the degeneracy."""
    ci, cj = _two_masked_contexts(use_positional_conv=True)
    diff = float((ci - cj).abs().max())
    assert diff > 1e-3, (
        f"positional conv did not distinguish two masked positions; "
        f"max |difference| = {diff:.3e}")

    # And the two are not merely numerically different but genuinely distinct.
    cos = F.cosine_similarity(ci.unsqueeze(0), cj.unsqueeze(0)).item()
    assert cos < 0.9999, f"contexts are near-collinear (cos = {cos:.6f})"


def test_positional_conv_preserves_sequence_length():
    """kernel 128 is even; SamePad must trim the extra frame."""
    conv = RelativePositionalConv(dim=64, kernel_size=128, groups=16)
    for T in (60, 128, 129, 500):
        x = torch.randn(2, T, 64)
        assert conv(x).shape == x.shape, (
            f"positional conv changed the sequence length at T={T}")


def test_positional_conv_uses_the_fairseq_geometry():
    """kernel 128, groups 16, weight-normalised, as in wav2vec 2.0 Base."""
    conv = RelativePositionalConv(dim=512, kernel_size=128, groups=16)
    assert conv.conv.kernel_size[0] == 128
    assert conv.conv.groups == 16
    names = dict(conv.conv.named_parameters())
    assert 'weight_g' in names and 'weight_v' in names, (
        "expected weight normalisation on the positional conv")


# ----------------------------------------------------------------------------
# 5. ENCODER WIRING
# ----------------------------------------------------------------------------
def test_frontend_defaults_to_supervised_parity_and_can_opt_into_fairseq():
    """The default frontend must stay byte-compatible with the supervised models.

    scripts/train_fsq.py and scripts/train_rvq_bpc.py pad their kernel-3 conv
    layers, so 16000 samples give 50 frames. The SSL encoder copies that stack
    on purpose: the FSQ-vs-RVQ comparison only holds if both sides see the same
    frontend. fairseq_padding=True opts into wav2vec 2.0's unpadded stack (49
    frames) for a fidelity ablation, and must stay opt-in.
    """
    wav = torch.randn(2, 16000)

    default = SSLEncoder(quantiser='fsq16d', num_transformer_layers=2).eval()
    fairseq = SSLEncoder(quantiser='fsq16d', num_transformer_layers=2,
                         fairseq_padding=True).eval()
    with torch.no_grad():
        assert default.encode_cnn(wav).shape[1] == 50, (
            "the default frontend no longer matches the supervised models: "
            "every FSQ-vs-RVQ comparison assumes a shared frontend")
        assert fairseq.encode_cnn(wav).shape[1] == 49, (
            "fairseq_padding=True should reproduce wav2vec 2.0's frame count")

    # Both are 320x downsampling (20 ms frames); only the edge effect differs.
    for model, expected in ((default, 100), (fairseq, 99)):
        with torch.no_grad():
            assert model.encode_cnn(torch.randn(1, 32000)).shape[1] == expected


def test_quantiser_sits_on_the_cnn_output():
    """The target branch must quantise z, not the Transformer output.

    Asserted structurally: the quantised target must be a function of the CNN
    output alone, so it is unchanged when the mask (which only affects the
    context branch) changes.
    """
    torch.manual_seed(5)
    model = SSLEncoder(quantiser='fsq16d', num_transformer_layers=2).eval()
    wav = torch.randn(2, 16000)

    with torch.no_grad():
        # Take the frame rate from the CNN rather than hardcoding it: this
        # stack pads the kernel-3 layers, so 16000 samples give 50 frames, not
        # the 49 of fairseq's zero-padded stack.
        T = model.encode_cnn(wav).shape[1]
        m1 = torch.zeros(2, T, dtype=torch.bool); m1[:, 5:15] = True
        o1 = model(wav, mask=m1)
        m2 = torch.zeros_like(o1['mask']); m2[:, 20:30] = True
        o2 = model(wav, mask=m2)

    assert torch.allclose(o1['target'], o2['target'], atol=1e-6), (
        "the quantised target changed when the mask changed: the quantiser is "
        "reading the masked/context branch, not the CNN output")
    assert not torch.allclose(o1['context'], o2['context'], atol=1e-4), (
        "the context did not change when the mask changed")


def test_fsq_branch_exposes_no_diversity_distribution():
    """FSQ has no learned codebook, so L_d has nothing to act on."""
    model = SSLEncoder(quantiser='fsq16d', num_transformer_layers=2).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 16000))
    assert out['quantiser_out']['diversity_probs'] is None, (
        "the FSQ branch should expose no diversity distribution (see "
        "losses/w2v2_contrastive.py for why this asymmetry is deliberate)")


def test_rvq_branch_exposes_a_usable_diversity_distribution():
    torch.manual_seed(6)
    model = SSLEncoder(quantiser='rvq4l', num_transformer_layers=2).eval()
    out = model(torch.randn(1, 16000))
    probs = out['quantiser_out']['diversity_probs']
    assert probs is not None and probs.dim() == 3
    assert probs.shape[1] == 4 and probs.shape[2] == 128
    assert torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1)), atol=1e-4)

    loss, stats = CodebookDiversityLoss(4, 128)(probs)
    loss.backward()
    assert torch.isfinite(loss) and 0.0 <= loss.item() <= 1.0
    grads = [p.grad for p in model.quantiser.parameters() if p.grad is not None]
    assert grads, "L_d produced no gradient on the RVQ codebook"


def test_end_to_end_step_runs_for_every_quantiser():
    """Each branch produces a finite loss and a gradient."""
    for name in ('rvq4l', 'fsq16d', 'gumbel_pq'):
        torch.manual_seed(19)
        model = SSLEncoder(quantiser=name, num_transformer_layers=2)
        out = model(torch.randn(2, 16000))
        loss, stats = W2V2ContrastiveLoss(num_negatives=10)(
            out['context'], out['target'], out['mask'])
        assert torch.isfinite(loss), f"{name}: non-finite loss"
        loss.backward()
        n_grad = sum(1 for p in model.parameters()
                     if p.grad is not None and float(p.grad.abs().sum()) > 0)
        assert n_grad > 0, f"{name}: no parameter received a gradient"


# ----------------------------------------------------------------------------
def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}\n        {exc}")
        except Exception as exc:
            # An exception is a failure of this test, not of the run: without
            # this, one bad shape aborts the file and the remaining tests are
            # silently never executed.
            failed += 1
            print(f"ERROR {name}\n        {type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(_main())
