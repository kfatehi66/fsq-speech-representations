#!/usr/bin/env python3
"""
Build a LibriSpeech manifest for one subset, in the format that the trainers
and the evaluator read (they use only "audio_path").

Manifest entry (JSON array of dicts):
  {
    "audio_path": "<split>/<speaker>/<chapter>/<utt>.flac",   # relative to LIBRISPEECH_AUDIO/LibriSpeech
    "text": "TRANSCRIPTION TEXT",
    "speaker_id": <int>,
    "chapter_id": <int>,
    "utterance_id": "<speaker>-<chapter>-<index>",
    "duration": <float seconds, 3dp>
  }

Usage:
  python scripts/build_split_manifest.py --split dev-other
  python scripts/build_split_manifest.py --split test-other
"""
import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from paths import LIBRISPEECH_AUDIO, MANIFESTS_DIR


def load_transcripts(split_root: Path) -> dict:
    """Build {utterance_id: text} from all .trans.txt files under a split."""
    transcripts = {}
    for trans_file in split_root.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                transcripts[utt_id] = text
    return transcripts


def build_manifest(split: str) -> list:
    split_root = LIBRISPEECH_AUDIO / "LibriSpeech" / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    transcripts = load_transcripts(split_root)
    audio_root = LIBRISPEECH_AUDIO / "LibriSpeech"

    entries = []
    missing_text = 0
    for flac_path in sorted(split_root.rglob("*.flac")):
        utt_id = flac_path.stem
        chapter_id = int(flac_path.parent.name)
        speaker_id = int(flac_path.parent.parent.name)

        if utt_id not in transcripts:
            print(f"Warning: no transcription for {utt_id}", file=sys.stderr)
            missing_text += 1
            continue

        info = sf.info(str(flac_path))
        duration = info.frames / info.samplerate

        rel_path = flac_path.relative_to(audio_root)

        entries.append({
            "audio_path": str(rel_path),
            "text": transcripts[utt_id],
            "speaker_id": speaker_id,
            "chapter_id": chapter_id,
            "utterance_id": utt_id,
            "duration": round(duration, 3),
        })

    entries.sort(key=lambda e: e["utterance_id"])
    if missing_text:
        print(f"Warning: {missing_text} flac files had no transcription (skipped)", file=sys.stderr)
    return entries


def main():
    parser = argparse.ArgumentParser(description="Build a LibriSpeech split manifest.")
    parser.add_argument("--split", required=True,
                        help="e.g. dev-other, test-other, dev-clean, test-clean, train-clean-100")
    parser.add_argument("--out", default=None,
                        help="Output JSON path (default: MANIFESTS_DIR/<split>.json)")
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else (MANIFESTS_DIR / f"{args.split}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = build_manifest(args.split)

    with open(out_path, "w") as f:
        json.dump(entries, f)

    print(f"Wrote {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
