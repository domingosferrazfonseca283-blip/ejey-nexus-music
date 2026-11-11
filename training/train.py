#!/usr/bin/env python3
"""Train a small decoder-only Transformer on symbolic MIDI event sequences."""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EventDataset(Dataset):
    def __init__(self, sequences, stoi, block_size):
        self.examples = []
        for sequence in sequences:
            ids = [stoi["<BOS>"]] + [stoi[token] for token in sequence] + [stoi["<EOS>"]]
            for start in range(0, max(1, len(ids) - 1), block_size):
                chunk = ids[start:start + block_size + 1]
                if len(chunk) >= 2:
                    self.examples.append(chunk)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        chunk = self.examples[index]
        x = chunk[:-1]
        y = chunk[1:]
        length = len(x)
        x = x + [0] * (MAX_BLOCK - len(x))
        y = y + [-100] * (MAX_BLOCK - len(y))
        return torch.tensor(x), torch.tensor(y)


MAX_BLOCK = 128


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, width=192, layers=4, heads=4):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(block_size, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, ids):
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device).unsqueeze(0)
        hidden = self.token(ids) + self.position(positions)
        mask = torch.triu(torch.ones(length, length, device=ids.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=mask)
        return self.output(self.norm(hidden))


def load_sequences(path: Path):
    sequences = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                item = json.loads(line)
                events = item.get("events")
                if isinstance(events, list) and len(events) >= 2 and all(isinstance(x, str) for x in events):
                    sequences.append(events)
            except json.JSONDecodeError as exc:
                print(f"Linha {line_number} ignorada: {exc}")
    if not sequences:
        raise SystemExit("Dataset vazio ou inválido. Executa primeiro prepare_dataset.py.")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sequences.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/ejey-midi-v1"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=MAX_BLOCK)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 8 <= args.block_size <= MAX_BLOCK:
        raise SystemExit("Parâmetros inválidos: epochs/batch-size >= 1 e block-size entre 8 e 128.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequences = load_sequences(args.data)
    tokens = sorted({token for sequence in sequences for token in sequence})
    vocab = ["<PAD>", "<BOS>", "<EOS>"] + [token for token in tokens if token not in {"<PAD>", "<BOS>", "<EOS>"}]
    stoi = {token: index for index, token in enumerate(vocab)}
    dataset = EventDataset(sequences, stoi, args.block_size)
    # EventDataset pads to MAX_BLOCK; model uses the selected block size.
    if args.block_size != MAX_BLOCK:
        raise SystemExit("Nesta versão usa --block-size 128 (limite atual do protótipo).")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = MusicTransformer(len(vocab), args.block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Sequências: {len(sequences)} | exemplos: {len(dataset)} | vocabulário: {len(vocab)} | dispositivo: {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        mean_loss = total_loss / max(1, batches)
        print(f"Época {epoch:03d}/{args.epochs} | loss={mean_loss:.4f}")
        torch.save({"model_state": model.state_dict(), "vocab_size": len(vocab),
                    "block_size": args.block_size, "epoch": epoch}, args.out / "model.pt")
    (args.out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    config = {"architecture": "decoder-only Transformer", "task": "symbolic MIDI event next-token prediction",
              "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
              "block_size": args.block_size, "vocab_size": len(vocab), "device": str(device),
              "training_sequences": len(sequences)}
    (args.out / "training_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Treino concluído. Checkpoint: {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
