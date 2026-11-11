#!/usr/bin/env python3
"""Generate a conditioned symbolic MIDI draft from an EJey checkpoint."""
import argparse
import json
from pathlib import Path

import pretty_midi
import torch

from train import MusicTransformer


def sample_next(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-4)
    logits[0] = float("-inf")
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits[logits < values[-1]] = float("-inf")
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise RuntimeError("Invalid sampling distribution.")
    return torch.multinomial(probabilities, 1).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/generated/ejey-draft.mid"))
    parser.add_argument("--genre", default="Other")
    parser.add_argument("--mood", default="Neutral")
    parser.add_argument("--tempo", default="Moderate")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--program", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.max_tokens < 8 or args.temperature <= 0 or args.top_k < 0 or not 0 <= args.program <= 127:
        raise SystemExit("Invalid generation parameters.")
    if not args.checkpoint.is_file() or not args.vocab.is_file():
        raise SystemExit("Checkpoint/vocabulary not found. Train the model first.")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocabulary = json.loads(args.vocab.read_text(encoding="utf-8"))
    token_to_id = {token: i for i, token in enumerate(vocabulary)}
    conditions = [f"GENRE:{args.genre}", f"MOOD:{args.mood}", f"TEMPO:{args.tempo}"]
    missing = [token for token in conditions if token not in token_to_id]
    if missing:
        raise SystemExit("Condition not found in training vocabulary: " + ", ".join(missing))
    config = checkpoint["model_config"]
    if config["vocab_size"] != len(vocabulary):
        raise SystemExit("Vocabulary does not match checkpoint.")
    model = MusicTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    ids = [token_to_id["<BOS>"]] + [token_to_id[token] for token in conditions]
    generated = []
    with torch.no_grad():
        for _ in range(args.max_tokens):
            x = torch.tensor([ids[-config["block_size"]:]], dtype=torch.long, device=device)
            next_id = sample_next(model(x)[0, -1].clone(), args.temperature, args.top_k)
            token = vocabulary[next_id]
            if token == "<EOS>":
                break
            ids.append(next_id)
            if token.startswith(("WAIT:", "NOTE:")):
                generated.append(token)
    if not generated:
        raise SystemExit("No MIDI events generated. Add more varied training sequences and retry.")

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=args.program)
    current_time = 0.0
    for token in generated:
        parts = token.split(":")
        try:
            if parts[0] == "WAIT" and len(parts) == 2:
                current_time += max(0, min(100, int(parts[1]))) / 10.0
            elif parts[0] == "NOTE" and len(parts) == 4:
                pitch, velocity, duration = map(int, parts[1:])
                if 0 <= pitch <= 127:
                    instrument.notes.append(pretty_midi.Note(
                        velocity=max(1, min(127, velocity * 16 + 8)),
                        pitch=pitch,
                        start=current_time,
                        end=current_time + max(1, min(80, duration)) / 10.0,
                    ))
        except ValueError:
            continue
    if not instrument.notes:
        raise SystemExit("Generated events contained no valid notes.")
    midi.instruments.append(instrument)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(args.output))
    args.output.with_suffix(".json").write_text(
        json.dumps({"conditions": {"genre": args.genre, "mood": args.mood, "tempo": args.tempo},
                    "events": generated}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"MIDI draft saved: {args.output} | notes: {len(instrument.notes)} | device: {device}")
    print("Symbolic MIDI is not rendered audio. Open the file in a DAW to audition and arrange it.")


if __name__ == "__main__":
    main()
