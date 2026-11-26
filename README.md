# Finite Scalar Quantisation for Speech Representations

Training and evaluation code for the paper "Finite Scalar Quantisation for
Speech Representations: Dimensionality Matters More Than Codebook Size"
(Computer Speech & Language). The models are frame-level phoneme classifiers
trained on LibriSpeech, with a finite scalar quantisation (FSQ) or residual
vector quantisation (RVQ) bottleneck, or no bottleneck, and a wav2vec 2.0-style
self-supervised model with an FSQ, RVQ or Gumbel product quantiser, evaluated
by frozen linear probes. For the results, see the paper.

## Contents

| Path | Purpose |
|---|---|
| `paths.py` | Locations of data, checkpoints and outputs; every location can be set with an environment variable |
| `data/phoneme_vocab.py` | Phoneme inventory: 39 CMU phonemes and 4 special tokens (43 classes) |
| `data/dataset.py` | Generic waveform dataset and the re-anchoring of manifest audio paths |
| `models/fsq.py` | Finite scalar quantiser, with an optional symmetry-preserving bound |
| `models/fsq_legacy.py` | Earlier FSQ implementation without the even-L offset; used only by the tests |
| `models/ssl_encoder.py` | Encoder for the self-supervised models (FSQ, RVQ or Gumbel product quantiser on the CNN output) |
| `models/checkpoint_loading.py`, `models/checkpoint_fingerprint.py` | Building a model from a checkpoint and checking it against the configuration stored in the checkpoint |
| `models/*.py` (other files) | Modules imported by `models/__init__.py` |
| `losses/w2v2_contrastive.py` | Contrastive and codebook-diversity losses of the self-supervised objective |
| `losses/*.py` (other files) | Modules imported by `losses/__init__.py` |
| `scripts/train_*.py` | Trainers (see Training) |
| `scripts/build_split_manifest.py` | Manifest of one LibriSpeech subset (see Data) |
| `scripts/make_untrained_ssl_checkpoint.py` | Untrained encoder checkpoint for the floor of the probes (see Linear probes) |
| `scripts/evaluate_fsq.py` | Evaluation of a supervised checkpoint (see Evaluation) |
| `scripts/rate_analysis.py` | Code-usage and entropy statistics written by the evaluator |
| `scripts/broad_phonetic_classes.py` | Mappings from phonemes to broad phonetic classes |
| `scripts/probe_frozen_linear.py` | Frozen linear phoneme probe of a self-supervised checkpoint |
| `scripts/probe_layerwise.py` | Frozen probes at every layer of a self-supervised checkpoint |
| `tests/` | Tests of the modules above |
| `docs/PREREGISTERED.md` | Registered decision rules |

## Requirements and installation

The cluster launch scripts (not part of this repository) activate an
environment with Python 3.12.13, as checked when the control arms were
launched; `requirements.txt` pins the versions installed there of the packages
that the code imports. The logs of the earlier runs all report PyTorch
2.7.1+cu118, a build for CUDA 11.8. torchaudio reads the FLAC files through an
audio back end, which in that environment was the `soundfile` package; the
manifest builder also imports it directly. Training and evaluation were run
with one GPU per job.

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

All commands below are run from the repository root with the repository on
`PYTHONPATH`, because the scripts import the top-level module `paths` and the
packages `data`, `models`, `losses` and `scripts` from the repository root:

```bash
export PYTHONPATH=$(pwd)
export PROTOSPEECH_ARTEFACTS_ROOT=/path/to/artefacts   # default: artefacts/ in the repository
```

## Data

The code uses five LibriSpeech subsets:

| Subset | Use |
|---|---|
| `train-clean-100` | training (supervised models, self-supervised pre-training and probes) |
| `dev-clean` | validation during training (selection of `best.pt`) and the main evaluation |
| `dev-other`, `test-clean`, `test-other` | held-out evaluation |

Expected layout under `$PROTOSPEECH_ARTEFACTS_ROOT`:

```
librispeech/LibriSpeech/<subset>/<speaker>/<chapter>/<utterance>.flac
alignments/<subset>/<speaker>/<chapter>/<utterance>.TextGrid
manifests/train-clean-100.json
manifests/dev-clean.json
manifests/dev-other.json
manifests/test-clean.json
manifests/test-other.json
checkpoints/
outputs/
```

Each location can be set separately with `PROTOSPEECH_LIBRISPEECH_AUDIO`
(the directory that contains `LibriSpeech/`), `PROTOSPEECH_ALIGNMENTS_DIR`,
`PROTOSPEECH_MANIFESTS_DIR`, `PROTOSPEECH_CHECKPOINTS_DIR` and
`PROTOSPEECH_OUTPUTS_DIR`. The trainers always read
`train-clean-100.json` and `dev-clean.json` from the manifest directory; the
evaluator reads the manifest given by `--manifest`.

**Manifests.** A manifest is a JSON list of objects with an `audio_path` field,
which is all that the trainers and the evaluator read. `audio_path` is either
relative to the `LibriSpeech/` directory or absolute, and must end in
`<subset>/<speaker>/<chapter>/<utterance>.flac` (or `.wav`), because the
alignment file is found from these four path components. Absolute paths
written on another machine can be re-rooted by setting
`PROTOSPEECH_MANIFEST_AUDIO_PREFIX` to the prefix that precedes `<subset>/`,
including its trailing slash. Utterances without an alignment file are
skipped. The manifests are not part of this repository. The order of their
entries matters: under a fixed seed the trainers' shuffled batches depend on
it, and the evaluator reads a manifest in file order (see Evaluation).

**Building a manifest.** `scripts/build_split_manifest.py` writes the manifest
of one subset:

```bash
python scripts/build_split_manifest.py --split dev-other
```

It lists every `.flac` file under `LibriSpeech/<subset>/` that has a line in the
subset's `.trans.txt` files, with `audio_path` relative to the `LibriSpeech/`
directory and the fields `text`, `speaker_id`, `chapter_id`, `utterance_id`
and `duration` (seconds, three decimals), sorted by utterance ID. It writes
`<subset>.json` to the manifest directory, or to the path given by `--out`, and
reports files without a transcription on standard error. In the runs of the
paper the dev-other and test-other manifests held relative paths of this form,
and the other three manifests held absolute paths written on an earlier
machine (see `PROTOSPEECH_MANIFEST_AUDIO_PREFIX` above).

**Frame-level phoneme labels.** The labels come from Montreal Forced Aligner
alignments (pretrained English MFA acoustic model and the CMU-based English
MFA dictionary) stored as Praat TextGrid files; the alignments themselves are
not part of this repository. The code reads the `phones` tier and maps each
label with `normalize_phoneme` (`data/phoneme_vocab.py`; the evaluator has an
identical copy): trailing stress digits 0 to 2 are removed and the label is
upper-cased; `SIL` and `SPN` become `<sil>`, and any other label outside the 39
phonemes becomes `<unk>`. The phones tier is therefore expected to use the 39
ARPAbet phonemes, with `sil`, `sp`, `spn` or an empty label for silence.
Frames are 20 ms (320 samples at 16 kHz). The evaluator, the probes and all trainers except the Concat and
Gated baselines drop the intervals labelled `sil`, `sp`, `spn` or empty,
assign each remaining interval to the frames from `int(xmin / 0.02)` to
`int(xmax / 0.02)`, end excluded, and give every other frame the ignore index
-100, which counts neither in the loss nor in the accuracy; a remaining
interval that becomes `<unk>` or `<sil>` is trained and scored as that class.
The Concat and
Gated trainers (`scripts/train_fusion_phoneme_supervised.py`,
`scripts/train_gated_supervised.py`) instead label each frame with the
interval that contains the frame's centre and train silence and unaligned
frames as the class `<sil>`; their checkpoints are evaluated like all others.
Audio is read at 16 kHz (resampled otherwise) and mixed to mono. Training and
validation crop utterances longer than 15 s to a random 15 s window; the
evaluator keeps the first 30 s. No utterance is excluded for its length: the
dataset classes of the trainers and the probes store a `min_audio_length`
argument but do not use it, and the evaluator's accepts one but neither stores
nor uses it.

## Training

In the commands below, `$SEED` is one of 42, 123, 456, 789 and 2024, and
`$CKPT` is the output directory of the run (for example
`$PROTOSPEECH_ARTEFACTS_ROOT/checkpoints/fsq_16d_seed$SEED`). Each configuration
in the tables of supervised configurations and control arms was trained with
each of these seeds; the runs under "Single runs without a fixed seed" were
trained once, without `--seed`. For example:

```bash
for SEED in 42 123 456 789 2024; do
  CKPT=$PROTOSPEECH_ARTEFACTS_ROOT/checkpoints/fsq_16d_seed$SEED
  python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" \
      --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"
done
```

The supervised trainers fix the remaining settings: batch size 8, AdamW with
weight decay 0.01, a one-cycle learning-rate schedule with 8% warm-up, gradient
clipping at 1.0, and validation on dev-clean every 2,000 steps. `best.pt` is the
checkpoint with the highest dev-clean accuracy at a validation step, measured
on the trainer's own labels (see Data): phoneme accuracy, except in
`scripts/train_rvq_bpc.py` (RVQ 4L×128 BPC-8, RVQ 8L×128 BPC-8, 2L BPC-8,
2L BPC-VUS and the RVQ sweep), which selects on broad-phonetic-class accuracy.
`final.pt` is the checkpoint after the last step.

### Supervised configurations

| Configuration | Command |
|---|---|
| FSQ mixed [8,5,5,5] (FSQ-4d) | `python scripts/train_fsq.py --fsq_levels "8,5,5,5" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ mixed [8,8,5,5,5] | `python scripts/train_fsq.py --fsq_levels "8,8,5,5,5" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^4 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^5 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^6 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^7 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^8 (FSQ-8d) | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^16 (FSQ-16d) | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^32 (FSQ-32d) | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [3]^64 (FSQ-64d) | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ binary [2]^10 asym | `python scripts/train_fsq.py --fsq_levels "2,2,2,2,2,2,2,2,2,2" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ binary [2]^10 sym (symmetry-preserving bound) | `python scripts/train_fsq_sym.py --fsq_levels "2,2,2,2,2,2,2,2,2,2" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| FSQ [8,5,5,5] + BPC-8 | `python scripts/train_fsq.py --fsq_levels "8,5,5,5" --bpc_supervision --bpc_type 8class --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| RVQ 4L×128 BPC-8 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 4 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| RVQ 8L×128 BPC-8 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 8 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| 2L supervised | `python scripts/train_rvq_supervised.py --num_vq_layers 2 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --seed $SEED --output_dir "$CKPT"` |
| 2L BPC-8 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 2 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| 2L BPC-VUS | `python scripts/train_rvq_bpc.py --bpc_type 3class_vus --num_vq_layers 2 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| 2L hierarchical-8 | `python scripts/train_rvq_hierarchical.py --bpc_type 8class --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --bpc_weight 1.0 --fine_weight 1.0 --seed $SEED --output_dir "$CKPT"` |
| 2L hierarchical-V | `python scripts/train_rvq_hierarchical.py --bpc_type 3class_vus --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --bpc_weight 1.0 --fine_weight 1.0 --seed $SEED --output_dir "$CKPT"` |
| Concat (3-layer fusion) | `python scripts/train_fusion_phoneme_supervised.py --extract_layers 4,5,6 --max_steps 30000 --lr 1e-4 --seed $SEED --output_dir "$CKPT"` |
| Gated | `python scripts/train_gated_supervised.py --max_steps 30000 --lr 1e-4 --seed $SEED --output_dir "$CKPT"` |
| Identity (no VQ) | `python scripts/train_identity_baseline.py --max_steps 30000 --lr 1e-4 --seed $SEED --output_dir "$CKPT"` |

**Names.** In the paper, "supervised" means trained on phoneme labels alone.
The rows RVQ 2L×128, RVQ 4L×128 and RVQ 8L×128 of Table 3 (dimensionality
scaling) are the models trained on phoneme labels alone with
`scripts/train_rvq_supervised.py`: 2L supervised above, and control arms A and
B below. The 4-layer RVQ model trained with the BPC-8 objective
(`scripts/train_rvq_bpc.py --bpc_type 8class`, five seeds) is "RVQ 4L×128
BPC-8" in Table 8 (all supervised configurations), "4L BPC-8" in Table 6 (BPC
ablation) and "RVQ-4L (5-seed mean)" in Table 5 (RVQ sweep); RVQ 8L×128 BPC-8
is "8L BPC-8" in Table 6. In this README the heading "Supervised
configurations" and the phrase "supervised trainers" cover every model trained
on labels, including those with the BPC-8 objective, as opposed to the
self-supervised runs.

`scripts/train_rvq_bpc.py` trains the RVQ model on broad phonetic classes
(`--bpc_type 8class` or `3class_vus`): its loss is the broad-class
cross-entropy plus the phoneme cross-entropy weighted by `--fine_weight` and
the VQ loss weighted by `--vq_weight`. RVQ 4L×128 BPC-8, RVQ 8L×128 BPC-8,
2L BPC-8, 2L BPC-VUS and the RVQ sweep are trained with it; 2L supervised and controls A and B train RVQ
on phoneme labels alone with `scripts/train_rvq_supervised.py`.
`scripts/train_fsq.py` with `--bpc_supervision` uses the same combination
without a VQ loss (FSQ [8,5,5,5] + BPC-8, Control C and the two single FSQ runs
with BPC-8); without it the loss is the phoneme cross-entropy alone and
`--fine_weight` has no effect. `scripts/train_fsq_sym.py` runs
`scripts/train_fsq.py` unchanged with the quantiser's symmetry-preserving
bound switched on, and then writes a configuration fingerprint into `best.pt`
and `final.pt`. Concat (3-layer fusion) concatenates Transformer layers 4, 5
and 6 (`--extract_layers 4,5,6`; the trainer's default is 2, 4 and 6) and is
trained on phoneme labels without a teacher model; Gated fuses the same three
layers with a learned per-frame gate.

### Control arms

| Configuration | Command |
|---|---|
| Control A: RVQ 4L×128 supervised | `python scripts/train_rvq_supervised.py --num_vq_layers 4 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --seed $SEED --output_dir "$CKPT"` |
| Control B: RVQ 8L×128 supervised | `python scripts/train_rvq_supervised.py --num_vq_layers 8 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --seed $SEED --output_dir "$CKPT"` |
| Control C: FSQ [3]^16 + BPC-8 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --bpc_supervision --bpc_type 8class --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |
| Control D: FSQ [3]^16, repetition | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --seed $SEED --output_dir "$CKPT"` |

The four control arms are defined in `docs/PREREGISTERED.md` (rule R7).

### Single runs without a fixed seed

| Configuration | Command |
|---|---|
| RVQ-L2 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 2 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --output_dir "$CKPT"` |
| RVQ-L6 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 6 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --output_dir "$CKPT"` |
| RVQ-L8 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 8 --num_codes 128 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --output_dir "$CKPT"` |
| RVQ-K64 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 4 --num_codes 64 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --output_dir "$CKPT"` |
| RVQ-K256 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 4 --num_codes 256 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --output_dir "$CKPT"` |
| RVQ-K512 | `python scripts/train_rvq_bpc.py --bpc_type 8class --num_vq_layers 4 --num_codes 512 --max_steps 30000 --lr 1e-4 --vq_weight 1.0 --fine_weight 0.5 --output_dir "$CKPT"` |
| FSQ-16d + BPC-8 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --bpc_supervision --bpc_type 8class --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --output_dir "$CKPT"` |
| FSQ-32d + BPC-8 | `python scripts/train_fsq.py --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --bpc_supervision --bpc_type 8class --max_steps 30000 --lr 1e-4 --fine_weight 0.5 --output_dir "$CKPT"` |

The six RVQ runs are the single runs of the RVQ sweep (Table 5), whose point
with 4 layers and 128 codes is RVQ 4L×128 BPC-8. All eight runs use the BPC-8 objective and were
trained once without `--seed`, so the cuDNN settings of "Seeds and determinism"
were not applied. Each was evaluated on dev-clean from its `best.pt` with the
model options given under Evaluation.

### Self-supervised pre-training

The self-supervised runs pre-train on train-clean-100 with a wav2vec 2.0-style
contrastive objective and used seed 42. Every 2,000 steps they compute the
contrastive loss on the first 50 batches (400 utterances) of dev-clean, and
`best.pt` is the checkpoint with the lowest such loss; the masks, the negatives
and the 15 s crops of this validation are drawn afresh each time, so the loss
is a random estimate. The masking places max(1, int(mask_prob × T /
mask_length)) spans of `--mask_length` frames, which may overlap, in an
utterance of T frames; with `--mask_prob 0.065 --mask_length 10` it masks at
most 6.5% of the frames of an utterance longer than about 3 s.
`--diversity_weight` (alpha in the table) is the weight
of the codebook-diversity loss; it has no effect with the FSQ quantiser, which
has no learned codebook. With the RVQ quantiser the loss has no codebook or
commitment term, and the quantiser's straight-through output passes no
gradient to the codebook, so the codebook is trained only through the
diversity loss; with `--diversity_weight 0` its gradient is zero and it
changes only through the weight decay of AdamW. `--checkpoint_every 10000`
also writes `step_<N>.pt` every 10,000 steps.

| Configuration | Command |
|---|---|
| FSQ [3]^16, 100k steps | `python -u scripts/train_ssl_w2v2.py --quantiser fsq16d --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --max_steps 100000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0.1 --seed 42 --checkpoint_every 10000 --output_dir "$CKPT"` |
| RVQ-4L×128, alpha = 0.01, 100k steps | `python -u scripts/train_ssl_w2v2.py --quantiser rvq4l --num_vq_layers 4 --num_codes 128 --max_steps 100000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0.01 --seed 42 --checkpoint_every 10000 --output_dir "$CKPT"` |
| RVQ-4L×128, alpha = 0, 100k steps | `python -u scripts/train_ssl_w2v2.py --quantiser rvq4l --num_vq_layers 4 --num_codes 128 --max_steps 100000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0 --seed 42 --checkpoint_every 10000 --output_dir "$CKPT"` |
| FSQ [3]^16, 20k steps | `python -u scripts/train_ssl_w2v2.py --quantiser fsq16d --fsq_levels "3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3" --max_steps 20000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0.1 --seed 42 --output_dir "$CKPT"` |
| RVQ-4L×128, alpha = 0.1, 20k steps | `python -u scripts/train_ssl_w2v2.py --quantiser rvq4l --num_vq_layers 4 --num_codes 128 --max_steps 20000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0.1 --seed 42 --output_dir "$CKPT"` |
| RVQ-4L×128, alpha = 0.01, 20k steps | `python -u scripts/train_ssl_w2v2.py --quantiser rvq4l --num_vq_layers 4 --num_codes 128 --max_steps 20000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0.01 --seed 42 --output_dir "$CKPT"` |
| RVQ-4L×128, alpha = 0, 20k steps | `python -u scripts/train_ssl_w2v2.py --quantiser rvq4l --num_vq_layers 4 --num_codes 128 --max_steps 20000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0 --seed 42 --output_dir "$CKPT"` |
| Gumbel product quantiser, 20k steps | `python -u scripts/train_ssl_w2v2.py --quantiser gumbel_pq --max_steps 20000 --lr 5e-4 --batch_size 8 --num_negatives 100 --temperature 0.1 --mask_prob 0.065 --mask_length 10 --diversity_weight 0.1 --seed 42 --output_dir "$CKPT"` |

The 20k-step runs are sensitivity screens; `docs/PREREGISTERED.md` (rule R4)
states what they may and may not be used for.

### Linear probes of the self-supervised models

Every pre-training run was evaluated with the frozen linear phoneme probe of
its `best.pt`. The layer-wise probes were run on `best.pt` of the three 100k
runs and of the 20k runs with the FSQ quantiser or with RVQ at alpha = 0.1,
0.01 and 0, and on `final.pt` of the two 100k RVQ runs. The probes train on
train-clean-100 and score dev-clean, with utterances longer than 15 s cropped
to a random 15 s window as in training. In the commands, `$OUT` is an output
directory and `$NAME` a label that is written into the output JSON:

```bash
python -u scripts/probe_frozen_linear.py --checkpoint "$CKPT/best.pt" \
    --max_steps 10000 --lr 1e-3 --seed 42 --output_json "$OUT/probe.json"
python -u scripts/probe_layerwise.py --checkpoint "$CKPT/best.pt" \
    --max_steps 10000 --lr 1e-3 --seed 42 --label "$NAME" --output_json "$OUT/layerwise.json"
```

The launch scripts also passed `--job_id` and `--commit_hash`, which are only
written into the output JSON.

**Untrained-encoder floor.** The floor of the probes was measured on an
encoder without pre-training. `scripts/make_untrained_ssl_checkpoint.py`
writes it as `best.pt` in `--output_dir`: a randomly initialised encoder,
seeded with `--seed` (default 42), with the RVQ quantiser (4 layers of 128
codes), which the probes do not read. The floor at the encoder output was
measured with the frozen linear probe on this checkpoint; the floor at the
best tap is the `best_accuracy` that the layer-wise probe reports for it, the
best of its 14 combinations of tap and head:

```bash
python scripts/make_untrained_ssl_checkpoint.py --output_dir "$CKPT"
python -u scripts/probe_frozen_linear.py --checkpoint "$CKPT/best.pt" \
    --max_steps 10000 --lr 1e-3 --seed 42 --output_json "$OUT/probe.json"
python -u scripts/probe_layerwise.py --checkpoint "$CKPT/best.pt" \
    --max_steps 10000 --lr 1e-3 --seed 42 --label untrained --output_json "$OUT/layerwise.json"
```

`probe_frozen_linear.py` reads the output of the context network after its
final layer normalisation. `probe_layerwise.py` probes the CNN output and each
of the six Transformer layers, each with a linear head and with a
layer-normalised linear head, all on the same frozen forward pass; its
`layer_6` is the last Transformer layer before that final normalisation.

The probe-versus-step trajectory of the two 100k runs FSQ [3]^16 and
RVQ-4L×128 with alpha = 0.01 probes `step_10000.pt` to `step_90000.pt` and
`final.pt` of each run at two taps:

```bash
python -u scripts/probe_layerwise.py --checkpoint "$CKPT/step_10000.pt" \
    --max_steps 10000 --lr 1e-3 --taps layer_1,layer_6 --heads linear \
    --seed 42 --label "$NAME" --output_json "$OUT/trajectory_step_10000.json"
```

## Evaluation

With `--checkpoint`, `scripts/evaluate_fsq.py` evaluates one checkpoint on one
manifest (without it, the script runs a fixed comparison of checkpoints named
in its code, which is not used here):

```bash
python scripts/evaluate_fsq.py --checkpoint "$CKPT/best.pt" \
    --manifest "$PROTOSPEECH_ARTEFACTS_ROOT/manifests/dev-clean.json" \
    <model options> --seed $SEED --output_json "$OUT/dev-clean_seed$SEED.json"
```

Here `$OUT` is an output directory. For the held-out subsets, replace
`dev-clean.json` by `dev-other.json`, `test-clean.json` or `test-other.json`.
The evaluator reads the manifest in file order, in batches of 8 utterances
zero-padded to the longest one, and the models apply no padding mask, so the
result for an utterance can depend on the other utterances in its batch. A
different order changes the batches, and with them the rate statistics, and can
change individual frame decisions. The model options of each configuration:

| Configuration | Model options |
|---|---|
| FSQ configurations without BPC, except [2]^10 sym; Control D | `--model_type fsq --fsq_levels "<levels of the configuration>"` |
| FSQ binary [2]^10 sym | `--model_type fsq --fsq_levels "2,2,2,2,2,2,2,2,2,2" --preserve_symmetry` |
| FSQ [8,5,5,5] + BPC-8; Control C; FSQ-16d + BPC-8; FSQ-32d + BPC-8 | `--model_type fsq_bpc --fsq_levels "<levels of the configuration>"` |
| RVQ 4L×128 BPC-8 | `--model_type bpc --num_vq_layers 4 --num_codes 128` |
| RVQ 8L×128 BPC-8 | `--model_type bpc --num_vq_layers 8 --num_codes 128` |
| RVQ sweep (RVQ-L2, RVQ-L6, RVQ-L8, RVQ-K64, RVQ-K256, RVQ-K512) | `--model_type bpc --num_vq_layers <layers> --num_codes <codes>` |
| 2L BPC-8 | `--model_type bpc --num_vq_layers 2 --num_codes 128 --num_bpc_classes 8` |
| 2L BPC-VUS | `--model_type bpc --num_vq_layers 2 --num_codes 128 --num_bpc_classes 3` |
| 2L supervised | `--model_type rvq --num_vq_layers 2 --num_codes 128` |
| Control A | `--model_type rvq --num_vq_layers 4 --num_codes 128` |
| Control B | `--model_type rvq --num_vq_layers 8 --num_codes 128` |
| 2L hierarchical-8 | `--model_type hierarchical --num_codes 128 --num_bpc_classes 8` |
| 2L hierarchical-V | `--model_type hierarchical --num_codes 128 --num_bpc_classes 3` |
| Concat (3-layer fusion) | `--model_type fusion --extract_layers 4,5,6` |
| Gated | `--model_type gated --extract_layers 4,5,6` |
| Identity (no VQ) | `--model_type identity` |

Checkpoints of `scripts/train_fsq.py` and `scripts/train_rvq_bpc.py` carry no
configuration fingerprint. For them the evaluator builds the model from the
options given, and a wrong value that leaves the shapes of the weights
unchanged (for example other FSQ levels with the same number of dimensions) is
not detected. The checkpoints of the other trainers carry one; the model is
then built from the fingerprint, and the options are used only for fields that
it does not record.

The held-out subsets were evaluated for FSQ [3]^16 (FSQ-16d), RVQ 4L×128 BPC-8
and Control A, for all five seeds.

**Output.** The evaluator writes one JSON file, to `--output_json` or, if that
is not given, to `single_eval_summary.json` in `--output_dir` (default
`fsq/single_eval` under the outputs directory, which is
`PROTOSPEECH_OUTPUTS_DIR` or `$PROTOSPEECH_ARTEFACTS_ROOT/outputs`; the script
creates `fsq/` there whenever it is imported). It holds:

- `overall_accuracy`: frame-level phoneme accuracy over all labelled frames, in
  per cent, including frames labelled `<unk>` (see Data), which appear in
  neither of the next two tables, and frames labelled `<sil>`, which appear in
  them as `<sil>` and as the class Silence;
- `phoneme`: for each phoneme, `accuracy` (a fraction), `correct`, `total` and
  its broad phonetic class `bpc`;
- `bpc`: for each broad phonetic class (Affricate, Diphthong, Fricative, Nasal,
  Plosive, Semi-vowel, Vowel, and Silence when frames are labelled `<sil>`),
  `accuracy`, `correct`, `total`, `frequency`,
  `error_count` and `error_contribution`, where `accuracy`, `frequency` and
  `error_contribution` are fractions. `accuracy` is the accuracy of the phoneme
  head over the frames whose reference phoneme belongs to the class; the
  broad-class head of the models trained with broad-class targets is not
  evaluated. Classes and phonemes without a labelled frame are left out;
- `rate_analysis`: code-usage and entropy statistics of the quantiser
  (`scripts/rate_analysis.py`), counted over every position of the
  zero-padded batches, padding included; for the unquantised models it is
  `null` and `rate_analysis_note` gives the reason. For FSQ [3]^64, whose
  joint code does not fit into a 64-bit integer (`joint_code_id` is
  `hash64`), the joint-code fields are computed from a hash whose coefficients
  are drawn again for every batch, so one code can be counted as several; the
  per-dimension fields are not affected. The masked rate values that
  `docs/PREREGISTERED.md` (rule R7) describes, and [3]^64 joint values computed
  with one fixed set of coefficients, were produced by a script that is not
  part of this repository;
- `load_provenance`: whether the checkpoint carried a configuration
  fingerprint, the configuration the model was built with, and the missing
  and unexpected state-dict keys;
- the values of `--checkpoint`, `--manifest`, `--model_type`, `--seed`,
  `--job_id`, `--eval_job_id` and `--commit_hash`.

The evaluator also logs the overall accuracy and the rate statistics (to
standard error) and prints the broad-class table, with the quantities of
`bpc`, to standard output.

## Seeds and determinism

Every trainer and both probes take `--seed`. When it is given, they seed
Python's `random`, NumPy and PyTorch (CPU and CUDA) and set
`torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False`. Without `--seed` neither is done. No
code sets `CUBLAS_WORKSPACE_CONFIG` or calls
`torch.use_deterministic_algorithms`, and apart from `PYTHONPATH`,
`PROTOSPEECH_PROJECT_ROOT` and `PROTOSPEECH_ARTEFACTS_ROOT` the launch scripts
set no environment variable that the code reads. Most launch scripts also
loaded CUDA 11.8 and cuDNN 8.7 modules of the cluster; the PyTorch build itself
provides CUDA 11.8 and cuDNN 9.1.0. The supervised configurations and the
control arms were trained with seeds 42, 123, 456, 789 and 2024, the single
runs without `--seed`, and the self-supervised runs and their probes with
seed 42.

The evaluator sets no seed: its `--seed`, `--job_id`, `--eval_job_id` and
`--commit_hash` options are only written into the output JSON. In the recorded
checks, evaluating the same checkpoint on GPUs of different families changed a
small number of frame decisions, while repeating an evaluation on the same GPU
family reproduced it exactly (`docs/PREREGISTERED.md`, rule R5). A seeded
training run repeated on other hardware or software need not reproduce its
earlier result exactly (rule R7, control D).

`best.pt` is selected on dev-clean, which is also the main evaluation subset.

## Registered decision rules

`docs/PREREGISTERED.md` lists the decision rules R1 to R7 with the date each
was fixed, when it fired and its outcome. R1 to R4 were written into it after
they had been fixed, from the project's records; R7 was entered before any of
its runs, and its outcome was added afterwards. The file says where a rule
survives only in paraphrase or was amended after it fired. It refers to launch
scripts, analysis scripts, output directories and internal records of the
project that are not part of this repository.

## Tests

```bash
python tests/test_checkpoint_fingerprint.py
python tests/test_checkpoint_loading.py
python tests/test_fsq_bound.py
python tests/test_probe_frame_alignment.py
python tests/test_rate_analysis.py
python tests/test_w2v2_contrastive.py
```

These need neither a GPU nor data. `tests/test_fsq_e2e_regression.py` needs a
trained FSQ [3]^16 (FSQ-16d) checkpoint (`--checkpoint`, default
`$PROTOSPEECH_ARTEFACTS_ROOT/checkpoints/fsq_16d_seed42/best.pt`) and the
dev-clean data; without the checkpoint it prints `SKIP` and exits with
status 0.

## Citation

K. Fatehi, E. Loweimi, S. Ouelha, N. Cummins. Finite Scalar Quantisation for
Speech Representations: Dimensionality Matters More Than Codebook Size.
Computer Speech & Language, accepted for publication, 2026.

## Contact

Kavan Fatehi, kavan.fatehi@kcl.ac.uk
