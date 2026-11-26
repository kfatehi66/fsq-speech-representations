"""
Central path configuration for ProtoSpeech.
All location-dependent constants live here, so that no script hard-codes a
path.

Override any path by setting the matching PROTOSPEECH_* env var. The defaults
are relative to the repository root (the directory that holds this file).
"""
import os
from pathlib import Path

# --- Roots ---
PROJECT_ROOT = Path(os.environ.get(
    "PROTOSPEECH_PROJECT_ROOT",
    str(Path(__file__).resolve().parent),
))
ARTEFACTS_ROOT = Path(os.environ.get(
    "PROTOSPEECH_ARTEFACTS_ROOT",
    str(Path(__file__).resolve().parent / "artefacts"),
))

# --- Data ---
MANIFESTS_DIR = Path(os.environ.get(
    "PROTOSPEECH_MANIFESTS_DIR",
    str(ARTEFACTS_ROOT / "manifests"),
))
ALIGNMENTS_DIR = Path(os.environ.get(
    "PROTOSPEECH_ALIGNMENTS_DIR",
    str(ARTEFACTS_ROOT / "alignments"),
))
LIBRISPEECH_AUDIO = Path(os.environ.get(
    "PROTOSPEECH_LIBRISPEECH_AUDIO",
    str(ARTEFACTS_ROOT / "librispeech"),
))
# Absolute prefix that data/dataset.py strips from manifest audio paths before
# joining them under LIBRISPEECH_AUDIO/LibriSpeech, for manifests that were
# written with absolute paths on another machine. Empty: nothing is stripped.
MANIFEST_AUDIO_PREFIX = os.environ.get("PROTOSPEECH_MANIFEST_AUDIO_PREFIX", "")

# --- Outputs ---
CHECKPOINTS_DIR = Path(os.environ.get(
    "PROTOSPEECH_CHECKPOINTS_DIR",
    str(ARTEFACTS_ROOT / "checkpoints"),
))
OUTPUTS_DIR = Path(os.environ.get(
    "PROTOSPEECH_OUTPUTS_DIR",
    str(ARTEFACTS_ROOT / "outputs"),
))
LOGS_DIR = Path(os.environ.get(
    "PROTOSPEECH_LOGS_DIR",
    str(ARTEFACTS_ROOT / "logs"),
))

# --- Python environment (for launch scripts) ---
VENV_PATH = Path(os.environ.get(
    "PROTOSPEECH_VENV_PATH",
    str(Path(__file__).resolve().parent / "venv"),
))

# --- Convenience: manifest files ---
MANIFEST_TRAIN_CLEAN_100 = MANIFESTS_DIR / "train-clean-100.json"
MANIFEST_DEV_CLEAN = MANIFESTS_DIR / "dev-clean.json"
MANIFEST_DEV_OTHER = MANIFESTS_DIR / "dev-other.json"
MANIFEST_TEST_CLEAN = MANIFESTS_DIR / "test-clean.json"
MANIFEST_TEST_OTHER = MANIFESTS_DIR / "test-other.json"
