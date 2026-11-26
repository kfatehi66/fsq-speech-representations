"""
wav2vec 2.0-faithful SSL encoder with a pluggable quantiser.

    waveform --> CNN feature encoder --> z  (B, T, 512)
                                         |
                          +--------------+--------------+
                          |                             |
                 target branch                  context branch
                 quantise z -> q                span-mask z, add relative
                 (rvq4l | fsq16d |              positional conv, run the
                  gumbel_pq)                    6-layer Transformer -> c
                          |                             |
                          +--------> contrastive loss <-+
                                     (losses/w2v2_contrastive.py)

Quantiser placement: the quantiser sits on the CNN output z, not on the
Transformer output. This is the wav2vec 2.0 arrangement, and it differs from
every supervised ProtoSpeech model, all of which quantise the Transformer
output. On the CNN output the quantiser builds prediction targets for a
self-supervised objective, whereas on the Transformer output it is an
information bottleneck in a supervised classifier. Quantising the Transformer
output here would make the context and the target the same computation and
the objective near-trivial.

Relative positional convolution: used here only. The supervised ProtoSpeech
models have no positional encoding of any kind. In the SSL setting, omitting
it does not merely weaken the model, it makes the objective degenerate:

    A Transformer encoder without positional information is permutation-
    equivariant. Every masked timestep of an utterance is replaced by the same
    learned mask embedding, so all masked positions present identical input to
    an encoder whose output depends only on (own input, multiset of all
    inputs). All masked positions in an utterance therefore produce identical
    context vectors c_t. The contrastive loss asks the model to match c_t to
    q_t while distinguishing it from distractors drawn from other masked
    timesteps of the same utterance. If every c_t is identical, no parameter
    setting can separate them, and the objective bottoms out at chance
    regardless of representation quality.

tests/test_w2v2_contrastive.py checks both halves of this directly (identical
contexts without the conv, distinct with it).

Following fairseq, the convolution is depthwise-grouped with kernel 128 and 16
groups, weight-normalised, GELU, and added as a residual to the masked features.

Diversity loss: used for rvq4l and gumbel_pq and, by design, not for fsq16d, which has no
learned codebook to collapse. See losses/w2v2_contrastive.py.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

QUANTISER_CHOICES = ('rvq4l', 'fsq16d', 'gumbel_pq')


# ---------------------------------------------------------------------------
# Relative positional convolution (fairseq wav2vec2 ConvolutionalPositional-
# Embedding)
# ---------------------------------------------------------------------------
class _SamePad(nn.Module):
    """Trim the trailing element when the kernel size is even.

    Conv1d with padding = k // 2 returns T + 1 frames for even k. fairseq drops
    the last one; without this the context sequence is one frame longer than
    the target sequence and the mask no longer aligns.
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        return x[:, :, :-self.remove] if self.remove > 0 else x


class RelativePositionalConv(nn.Module):
    """Grouped weight-normalised conv giving the encoder relative position info.

    Args:
        dim: channel width (512).
        kernel_size: 128, as in wav2vec 2.0 Base.
        groups: 16, as in wav2vec 2.0 Base.
    """

    def __init__(self, dim: int = 512, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=groups)
        # fairseq's initialisation for this layer.
        std = math.sqrt(4.0 / (kernel_size * dim))
        nn.init.normal_(self.conv.weight, mean=0.0, std=std)
        nn.init.constant_(self.conv.bias, 0.0)
        self.conv = nn.utils.weight_norm(self.conv, name='weight', dim=2)
        self.post = nn.Sequential(_SamePad(kernel_size), nn.GELU())

    def forward(self, x):
        """x: (B, T, D) -> (B, T, D). Residual, as in wav2vec 2.0."""
        y = self.post(self.conv(x.transpose(1, 2))).transpose(1, 2)
        return x + y


# ---------------------------------------------------------------------------
# Quantisers
# ---------------------------------------------------------------------------
class GumbelProductQuantiser(nn.Module):
    """wav2vec 2.0's own quantiser: G groups x V entries, Gumbel-softmax.

    G=2, V=320 as in the paper. Each group contributes D/G dimensions; the
    concatenation is projected back to D.

    Straight-through hard selection at the forward pass, soft probabilities for
    the gradient and for the diversity loss.
    """

    def __init__(self, dim: int = 512, num_groups: int = 2, num_entries: int = 320,
                 temp_max: float = 2.0, temp_min: float = 0.5,
                 temp_decay: float = 0.999995):
        super().__init__()
        if dim % num_groups != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_groups ({num_groups}): "
                f"the G group vectors are concatenated back to dim")
        self.num_groups = num_groups
        self.num_entries = num_entries
        self.entry_dim = dim // num_groups

        self.weight_proj = nn.Linear(dim, num_groups * num_entries)
        self.codebook = nn.Parameter(
            torch.FloatTensor(1, num_groups * num_entries, self.entry_dim).uniform_(-1, 1))
        self.out_proj = nn.Linear(dim, dim)

        self.temp_max, self.temp_min, self.temp_decay = temp_max, temp_min, temp_decay
        self.register_buffer('curr_temp', torch.tensor(float(temp_max)))

    def set_num_updates(self, step: int):
        """Anneal the Gumbel temperature, as in the paper."""
        t = max(self.temp_max * (self.temp_decay ** step), self.temp_min)
        self.curr_temp.fill_(t)

    def forward(self, z):
        B, T, _ = z.shape
        logits = self.weight_proj(z).view(B * T * self.num_groups, self.num_entries)

        if self.training:
            hard = F.gumbel_softmax(logits.float(), tau=float(self.curr_temp),
                                    hard=True).type_as(z)
        else:
            idx = logits.argmax(dim=-1)
            hard = F.one_hot(idx, self.num_entries).type_as(z)

        # Soft probabilities for L_d (gradient flows through these).
        probs = torch.softmax(logits.float(), dim=-1).type_as(z)
        probs = probs.view(B * T, self.num_groups, self.num_entries)

        # Sum within a group (one-hot, so this selects), then concatenate across
        # groups: G x entry_dim = dim. A single bmm over all G*V codebook rows
        # would instead sum across groups and return entry_dim = dim/G.
        sel = hard.view(B * T, self.num_groups, self.num_entries)
        codebook = self.codebook.view(self.num_groups, self.num_entries, self.entry_dim)
        q = torch.einsum('ngv,gvd->ngd', sel, codebook)
        q = q.reshape(B, T, self.num_groups * self.entry_dim)
        q = self.out_proj(q)

        indices = hard.view(B * T, self.num_groups, self.num_entries).argmax(-1)
        indices = indices.view(B, T, self.num_groups)

        return {
            'quantized': q,
            'diversity_probs': probs,
            'indices': [indices[..., g] for g in range(self.num_groups)],
            'num_groups': self.num_groups,
            'num_entries': self.num_entries,
        }


class _SoftProbRVQ(nn.Module):
    """RVQ-4L x 128 wrapper exposing soft assignment probabilities for L_d.

    The residual stack itself is imported from scripts/train_rvq_bpc.py rather
    than reimplemented, so the SSL branch quantises with exactly the same code
    as the supervised RVQ runs.

    Targets use the module's own hard argmin assignment. The soft distribution
    returned here is a separate softmax(-distance / tau) over each layer's
    codebook, used only by the diversity loss: hard argmin carries no
    gradient, so without this L_d would be a monitor rather than a regulariser.
    """

    def __init__(self, dim: int = 512, num_layers: int = 4, num_codes: int = 128,
                 tau: float = 1.0):
        super().__init__()
        from scripts.train_rvq_bpc import ResidualVQ  # deferred: pulls torchaudio
        self.rvq = ResidualVQ(num_layers=num_layers, num_codes=num_codes,
                              code_dim=dim)
        self.num_layers = num_layers
        self.num_codes = num_codes
        self.tau = tau

    def _soft_probs(self, z):
        """Per-layer softmax(-d^2/tau) over each codebook, along the residual."""
        probs, residual = [], z
        for layer in self.rvq.vq_layers:
            flat = residual.reshape(-1, residual.shape[-1])
            d = (flat.pow(2).sum(1, keepdim=True)
                 + layer.codebook.weight.pow(2).sum(1)
                 - 2 * flat @ layer.codebook.weight.t())
            probs.append(torch.softmax(-d / self.tau, dim=-1))
            with torch.no_grad():
                q = layer.codebook(d.argmin(dim=1)).view_as(residual)
            residual = residual - q
        return torch.stack(probs, dim=1)          # (B*T, L, K)

    def forward(self, z):
        out = self.rvq(z)
        return {
            'quantized': out['quantized'],
            'vq_loss': out.get('total_loss', out.get('vq_loss', 0.0)),
            'diversity_probs': self._soft_probs(z),
            'indices': out['all_indices'],
            'num_groups': self.num_layers,
            'num_entries': self.num_codes,
        }


class _FSQWrapper(nn.Module):
    """FSQ-16d ([3]^16) wrapper matching the quantiser interface.

    `diversity_probs` is None: FSQ has no learned codebook, so there is no
    distribution for L_d to act on and no collapse mode for it to prevent. The
    training loop therefore skips L_d for this branch (see the module
    docstring).
    """

    def __init__(self, dim: int = 512, levels=None, preserve_symmetry: bool = False):
        super().__init__()
        from models.fsq import FSQ
        self.levels = list(levels) if levels else [3] * 16
        self.fsq = FSQ(input_dim=dim, levels=self.levels,
                       preserve_symmetry=preserve_symmetry)

    def forward(self, z):
        out = self.fsq(z)
        return {
            'quantized': out['quantized'],
            'vq_loss': out['vq_loss'],
            'diversity_probs': None,      # no learned codebook (see docstring)
            'indices': out['all_indices'],
            'codes': out['codes'],
            'num_groups': len(self.levels),
            'num_entries': max(self.levels),
        }


def build_quantiser(name: str, dim: int = 512, **kw):
    if name == 'rvq4l':
        return _SoftProbRVQ(dim=dim, num_layers=kw.pop('num_vq_layers', 4),
                            num_codes=kw.pop('num_codes', 128))
    if name == 'fsq16d':
        return _FSQWrapper(dim=dim, levels=kw.pop('fsq_levels', [3] * 16),
                           preserve_symmetry=kw.pop('preserve_symmetry', False))
    if name == 'gumbel_pq':
        return GumbelProductQuantiser(dim=dim,
                                      num_groups=kw.pop('gumbel_groups', 2),
                                      num_entries=kw.pop('gumbel_entries', 320))
    raise ValueError(f"Unknown quantiser {name!r}; expected one of {QUANTISER_CHOICES}")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class SSLEncoder(nn.Module):
    """CNN -> {quantised target, masked+positional Transformer context}.

    Args:
        quantiser: one of QUANTISER_CHOICES.
        use_positional_conv: leave True. Exposed solely so that
            tests/test_w2v2_contrastive.py can demonstrate the degeneracy the
            module docstring describes; setting it False in a real run makes
            the contrastive objective unlearnable by construction.
        proj_dim: shared space the context and target are compared in (256).
    """

    def __init__(self, quantiser: str = 'rvq4l', hidden_dim: int = 512,
                 num_transformer_layers: int = 6, num_heads: int = 8,
                 dropout: float = 0.1, proj_dim: int = 256,
                 use_positional_conv: bool = True,
                 pos_conv_kernel: int = 128, pos_conv_groups: int = 16,
                 fairseq_padding: bool = False,
                 **quantiser_kw):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.quantiser_name = quantiser
        self.use_positional_conv = use_positional_conv
        self.fairseq_padding = fairseq_padding

        # CNN feature encoder. By default this is the same stack as in every
        # supervised ProtoSpeech model, so the SSL and supervised results share
        # a frontend, which keeps the two comparable.
        #
        # fairseq_padding=True zero-pads nothing on the kernel-3 layers, as in
        # wav2vec 2.0 proper. Both variants downsample by 320 (20 ms frames);
        # they differ only in the edge effect, so 16000 samples give 49 frames
        # instead of 50. Opt in for a fidelity ablation, at the cost of the
        # frontend parity above.
        pad3 = 0 if fairseq_padding else 1
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=pad3),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=pad3),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=pad3),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=pad3),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(512, 512), nn.GELU(),
            nn.Conv1d(512, hidden_dim, kernel_size=2, stride=2, padding=0),
            nn.GroupNorm(hidden_dim, hidden_dim), nn.GELU(),
        )

        # Learned mask embedding.
        self.mask_emb = nn.Parameter(torch.FloatTensor(hidden_dim).uniform_(-0.1, 0.1))

        self.pos_conv = (RelativePositionalConv(hidden_dim, pos_conv_kernel,
                                                pos_conv_groups)
                         if use_positional_conv else None)
        self.pre_transformer_norm = nn.LayerNorm(hidden_dim)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout,
                activation='gelu', batch_first=True, norm_first=True
            ) for _ in range(num_transformer_layers)
        ])
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Quantiser on the CNN output: the target branch.
        self.quantiser = build_quantiser(quantiser, dim=hidden_dim, **quantiser_kw)

        # Compare context and target in a shared lower-dimensional space.
        self.project_context = nn.Linear(hidden_dim, proj_dim)
        self.project_target = nn.Linear(hidden_dim, proj_dim)

        n = sum(p.numel() for p in self.parameters())
        logger.info(f"SSLEncoder: {n/1e6:.2f}M parameters")
        logger.info(f"  quantiser: {quantiser} (on the CNN output z)")
        logger.info(f"  positional conv: {use_positional_conv} "
                    f"(kernel {pos_conv_kernel}, groups {pos_conv_groups})")
        frontend = ('fairseq (padding 0)' if fairseq_padding
                    else 'ProtoSpeech-supervised parity (padding 1)')
        logger.info(f"  frontend: {frontend}")
        if not use_positional_conv:
            logger.warning(
                "  positional conv disabled: the encoder is permutation-"
                "equivariant, every masked position yields the same context "
                "vector, and the contrastive objective is degenerate. This is "
                "a test-only configuration.")

    def encode_cnn(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        return self.feature_encoder(waveform).transpose(1, 2)     # (B, T, D)

    def apply_mask(self, z, mask):
        """Replace masked timesteps with the learned mask embedding."""
        return torch.where(mask.unsqueeze(-1), self.mask_emb.to(z.dtype), z)

    def context(self, z_masked):
        h = z_masked
        if self.pos_conv is not None:
            h = self.pos_conv(h)
        h = self.pre_transformer_norm(h)
        for layer in self.transformer_layers:
            h = layer(h)
        return self.layer_norm(h)

    def forward(self, waveform, mask=None, mask_prob: float = 0.065,
                mask_length: int = 10, generator: torch.Generator = None):
        """
        Returns a dict with:
            context / target: (B, T, proj_dim), the two branches projected into
                the shared comparison space.
            mask: (B, T) bool.
            quantiser_out: the raw quantiser dict (diversity_probs, indices...).
        """
        from losses.w2v2_contrastive import compute_span_mask

        z = self.encode_cnn(waveform)                             # (B, T, D)
        B, T, _ = z.shape

        if mask is None:
            mask = compute_span_mask((B, T), mask_prob, mask_length,
                                     device=z.device, generator=generator)

        # Target branch: quantise the unmasked CNN output.
        q_out = self.quantiser(z)

        # Context branch: mask, then contextualise.
        c = self.context(self.apply_mask(z, mask))

        return {
            'context': self.project_context(c),
            'target': self.project_target(q_out['quantized']),
            'context_raw': c,
            'z': z,
            'mask': mask,
            'quantiser_out': q_out,
        }
