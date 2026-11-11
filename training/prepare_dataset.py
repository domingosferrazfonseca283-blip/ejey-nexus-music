#!/usr/bin/env python3
"""Convert MIDI files into licensed, condition-labelled EJey symbolic sequences."""
import argparse
import json
from pathlib import Path

import pretty_midi


def quantize_duration(seconds: float) -> int:
    return max(1, min(80, int(round(seconds * 10))))


def midi_to_events(path: Path) -> list[str]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            start = max(0.0, float(note.start))
            end = max(start + 0.1, float(note.end))
            notes.append((start, int(note.pitch), int(note.velocity), end - start))
    notes.sort(key=lambda item: (item[0], item[1]))
    events = []
    previous_start = 0.0
    for start, pitch, velocity, duration in notes:
        delta = max(0, min(100, int(round((start - previous_start) * 10))))
        events.append(f"WAIT:{delta}")
        events.append(f"NOTE:{pitch}:{max(1, min(8, velocity // 16))}:{quantize_duration(duration)}")
        previous_start = start
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/sequences.jsonl"))
    parser.add_argument("--genre", default="Other")
    parser.add_argument("--mood", default="Neutral")
    parser.add_argument("--tempo", default="Moderate")
    parser.add_argument("--license-label", default="user-confirmed")
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input directory not found: {args.input}")
    if not args.license_label.strip():
        raise SystemExit("A non-empty license label is required.")
    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in {".mid", ".midi"})
    if not files:
        raise SystemExit(f"No MIDI files found in {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    with args.output.open("w", encoding="utf-8") as target:
        for midi_path in files:
            try:
                events = midi_to_events(midi_path)
                if len(events) < 4:
                    skipped += 1
                    continue
                row = {
                    "source": midi_path.name,
                    "license_label": args.license_label.strip(),
                    "genre": args.genre.strip()[:60],
                    "mood": args.mood.strip()[:60],
                    "tempo": args.tempo.strip()[:40],
                    "events": events,
                }
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"SKIPPED {midi_path}: {exc}")
    print(f"MIDI found: {len(files)} | sequences saved: {written} | skipped: {skipped}")
    print(f"Dataset: {args.output}")
    print("Use only files you created, public-domain files, or files whose license explicitly permits model training.")


if __name__ == "__main__":
    main()
