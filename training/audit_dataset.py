#!/usr/bin/env python3
"""Audit a MIDI training folder and write a machine-readable dataset report."""
import argparse
import hashlib
import json
from pathlib import Path

import pretty_midi


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/dataset_audit.json"))
    parser.add_argument("--license-label", default="user-confirmed")
    args = parser.parse_args()
    if not args.input.is_dir():
        raise SystemExit(f"Input directory not found: {args.input}")
    if not args.license_label.strip():
        raise SystemExit("Provide a non-empty license label. This is a record, not legal verification.")

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in {".mid", ".midi"})
    seen = {}
    entries = []
    totals = {"files": len(files), "valid": 0, "invalid": 0, "duplicate_files": 0,
              "tracks": 0, "notes": 0, "duration_seconds": 0.0}
    for path in files:
        item = {"file": path.relative_to(args.input).as_posix(), "license_label": args.license_label.strip()}
        try:
            checksum = digest(path)
            item["sha256"] = checksum
            if checksum in seen:
                item["duplicate_of"] = seen[checksum]
                totals["duplicate_files"] += 1
            else:
                seen[checksum] = item["file"]
            midi = pretty_midi.PrettyMIDI(str(path))
            note_count = sum(len(track.notes) for track in midi.instruments)
            duration = float(midi.get_end_time())
            item.update({"tracks": len(midi.instruments), "notes": note_count, "duration_seconds": round(duration, 3)})
            if note_count == 0:
                raise ValueError("No note events found")
            totals["valid"] += 1
            totals["tracks"] += len(midi.instruments)
            totals["notes"] += note_count
            totals["duration_seconds"] += duration
        except Exception as error:
            item["error"] = str(error)
            totals["invalid"] += 1
        entries.append(item)

    report = {"report_version": 1, "license_label": args.license_label.strip(),
              "license_notice": "Labels are user-supplied records and do not verify ownership or permission.",
              "totals": {**totals, "duration_seconds": round(totals["duration_seconds"], 3)},
              "files": entries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    print(f"Audit report: {args.output}")
    if not files:
        raise SystemExit("No MIDI files found.")
    if totals["valid"] == 0:
        raise SystemExit("No valid MIDI files found. See the audit report.")


if __name__ == "__main__":
    main()
