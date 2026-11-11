#!/usr/bin/env python3
"""Convert licensed MIDI files into note-event sequences for EJey experiments."""
import argparse
import json
from pathlib import Path

import pretty_midi


def quantize_duration(seconds: float) -> int:
    # 100 ms bins, clipped to a useful symbolic range.
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
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"Pasta de entrada não encontrada: {args.input}")
    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in {".mid", ".midi"})
    if not files:
        raise SystemExit(f"Nenhum ficheiro MIDI encontrado em {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as target:
        for midi_path in files:
            try:
                events = midi_to_events(midi_path)
                if len(events) < 4:
                    skipped += 1
                    continue
                target.write(json.dumps({"source": midi_path.name, "events": events}, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                skipped += 1
                print(f"IGNORADO {midi_path}: {exc}")
    print(f"MIDI encontrados: {len(files)} | sequências guardadas: {written} | ignorados: {skipped}")
    print(f"Dataset: {args.output}")


if __name__ == "__main__":
    main()
