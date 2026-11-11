#!/usr/bin/env python3
"""Generate a MIDI sample from an EJey Music Transformer checkpoint."""
import argparse
import json
from pathlib import Path

import pretty_midi
import torch

from train import MusicTransformer


def sample_next(logits, temperature: float, top_k: int):
    logits = logits / temperature
    logits[0] = float("-inf")
    logits[1] = float("-inf")
    if top_k > 0:
        count = min(top_k, logits.numel())
        threshold = torch.topk(logits, count).values[-1]
        logits[logits < threshold] = float("-inf")
    probabilities = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probabilities).all() or probabilities.sum() <= 0:
        raise RuntimeError("Distribuição inválida durante a amostragem.")
    return int(torch.multinomial(probabilities, num_samples=1).item())


def events_to_midi(events, output: Path, program: int):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=max(0, min(127, program)))
    current_time = 0.0
    for token in events:
        parts = token.split(":")
        if len(parts) == 2 and parts[0] == "WAIT":
            try:
                current_time += max(0, min(100, int(parts[1]))) / 10.0
            except ValueError:
                continue
        elif len(parts) == 4 and parts[0] == "NOTE":
            try:
                pitch = max(0, min(127, int(parts[1])))
                velocity_bin = max(1, min(8, int(parts[2])))
                duration_bin = max(1, min(80, int(parts[3])))
            except ValueError:
                continue
            instrument.notes.append(pretty_midi.Note(
                velocity=max(1, min(127, velocity_bin * 16 + 8)),
                pitch=pitch,
                start=current_time,
                end=current_time + duration_bin / 10.0,
            ))
    if not instrument.notes:
        raise RuntimeError("Não foram geradas notas MIDI válidas. Treina mais ou altera a seed.")
    midi.instruments.append(instrument)
    output.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/generated/ejey-sample.mid"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--program", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.temperature <= 0 or args.top_k < 0:
        raise SystemExit("max-tokens >= 2, temperature > 0 e top-k >= 0 são obrigatórios.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.checkpoint.is_file() or not args.vocab.is_file():
        raise SystemExit("Checkpoint/vocabulário não encontrados. Treina primeiro o modelo.")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    vocabulary = json.loads(args.vocab.read_text(encoding="utf-8"))
    if not isinstance(vocabulary, list) or len(vocabulary) < 4:
        raise SystemExit("Vocabulário inválido.")
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    if not {"<PAD>", "<BOS>", "<EOS>"}.issubset(token_to_id):
        raise SystemExit("Faltam tokens especiais no vocabulário.")
    config = checkpoint["model_config"]
    if config["vocab_size"] != len(vocabulary):
        raise SystemExit("O vocabulário não corresponde ao checkpoint.")

    model = MusicTransformer(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    generated = []
    context = [token_to_id["<BOS>"]]
    with torch.no_grad():
        for _ in range(args.max_tokens):
            input_ids = torch.tensor(
                [context[-config["block_size"]:]], dtype=torch.long, device=device
            )
            logits = model(input_ids)[0, -1].clone()
            next_id = sample_next(logits, args.temperature, args.top_k)
            token = vocabulary[next_id]
            if token == "<EOS>":
                break
            generated.append(token)
            context.append(next_id)

    events_to_midi(generated, args.output, args.program)
    events_path = args.output.with_suffix(".json")
    events_path.write_text(
        json.dumps({"events": generated, "checkpoint": str(args.checkpoint)}, indent=2),
        encoding="utf-8",
    )
    print(f"MIDI gerado: {args.output}")
    print(f"Eventos guardados: {events_path}")
    print(f"Dispositivo: {device} | eventos: {len(generated)}")


if __name__ == "__main__":
    main()
