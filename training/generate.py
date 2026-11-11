#!/usr/bin/env python3
"""Sample symbolic MIDI events from a trained EJey Music Transformer."""
import argparse
import json
import random
from pathlib import Path

import pretty_midi
import torch

from train import MusicTransformer


def sample_next(logits, temperature, top_k):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        count = min(top_k, logits.numel())
        threshold = torch.topk(logits, count).values[-1]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, num_samples=1).item())


def events_to_midi(events, destination):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="EJey generated instrument")
    current_time = 0.0
    for event in events:
        parts = event.split(":")
        if parts[0] == "WAIT" and len(parts) == 2:
            try:
                current_time += max(0, min(100, int(parts[1]))) / 10.0
            except ValueError:
                continue
        elif parts[0] == "NOTE" and len(parts) == 4:
            try:
                pitch = max(0, min(127, int(parts[1])))
                velocity_bin = max(1, min(8, int(parts[2])))
                duration_bin = max(1, min(80, int(parts[3])))
            except ValueError:
                continue
            velocity = max(1, min(127, velocity_bin * 16 + 8))
            start = current_time
            end = start + duration_bin / 10.0
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity, pitch=pitch, start=start, end=end
                )
            )
    if not instrument.notes:
        raise SystemExit("A amostra não produziu notas MIDI válidas. Tenta outra seed.")
    midi.instruments.append(instrument)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ejey-music-v1/best.pt"))
    parser.add_argument("--vocab", type=Path, default=Path("runs/ejey-music-v1/vocab.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/generated/ejey-sample.mid"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if args.max_tokens < 1 or args.temperature <= 0 or args.top_k < 0:
        raise SystemExit("max-tokens/top-k/temperature inválidos.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not args.checkpoint.is_file() or not args.vocab.is_file():
        raise SystemExit("Checkpoint/vocabulário não encontrados. Treina primeiro o modelo.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocabulary = json.loads(args.vocab.read_text(encoding="utf-8"))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    if "<BOS>" not in token_to_id or "<EOS>" not in token_to_id:
        raise SystemExit("Vocabulário incompatível: faltam tokens especiais.")

    model = MusicTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    generated = []
    context = [token_to_id["<BOS>"]]
    with torch.no_grad():
        for _ in range(args.max_tokens):
            input_ids = torch.tensor(
                [context[-model.block_size:]], dtype=torch.long, device=device
            )
            logits = model(input_ids)[0, -1]
            next_id = sample_next(logits, args.temperature, args.top_k)
            token = vocabulary[next_id]
            if token == "<EOS>":
                break
            if token not in {"<PAD>", "<BOS>"}:
                generated.append(token)
            context.append(next_id)

    events_path = args.output.with_suffix(".json")
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(
        json.dumps({"events": generated, "seed": args.seed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    events_to_midi(generated, args.output)
    print(f"Eventos gerados: {len(generated)}")
    print(f"MIDI: {args.output}")
    print(f"Eventos: {events_path}")


if __name__ == "__main__":
    main()
