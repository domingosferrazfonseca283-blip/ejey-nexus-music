#!/usr/bin/env python3
"""Train the EJey conditional symbolic-music Transformer on licensed MIDI events."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]


def read_sequences(path):
    sequences = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Line {line_number} skipped: {error}")
                continue
            events = row.get("events") if isinstance(row, dict) else None
            if not isinstance(events, list) or len(events) < 2 or not all(isinstance(x, str) for x in events):
                continue
            metadata = [
                "GENRE:" + str(row.get("genre", "Other"))[:60],
                "MOOD:" + str(row.get("mood", "Neutral"))[:60],
                "TEMPO:" + str(row.get("tempo", "Moderate"))[:40],
            ]
            sequences.append(metadata + events)
    if not sequences:
        raise SystemExit("Dataset is empty or invalid. Run prepare_dataset.py first.")
    return sequences


class EventDataset(Dataset):
    def __init__(self, sequences, token_to_id, block_size):
        self.examples = []
        for sequence in sequences:
            ids = [token_to_id["<BOS>"]] + [token_to_id[token] for token in sequence] + [token_to_id["<EOS>"]]
            for start in range(0, len(ids) - 1, block_size):
                part = ids[start:start + block_size + 1]
                if len(part) > 1:
                    inputs, targets = part[:-1], part[1:]
                    pad = block_size - len(inputs)
                    self.examples.append((
                        inputs + [token_to_id["<PAD>"]] * pad,
                        targets + [-100] * pad,
                    ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        x, y = self.examples[index]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, width=192, layers=4, heads=4, dropout=0.1):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, token_ids):
        _, length = token_ids.shape
        if length > self.block_size:
            raise ValueError(f"sequence exceeds block_size={self.block_size}")
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(torch.ones(length, length, device=token_ids.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=causal_mask, src_key_padding_mask=token_ids.eq(0))
        return self.output(self.norm(hidden))


def mean_loss(model, loader, criterion, device):
    model.eval()
    total = batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            batches += 1
    return total / max(1, batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sequences.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/ejey-music-v0.2"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.block_size < 16:
        raise SystemExit("epochs/batch-size must be positive and block-size >= 16.")
    if args.width < 32 or args.layers < 1 or args.heads < 1 or args.width % args.heads:
        raise SystemExit("Invalid width/layers/heads configuration.")
    if not 0 <= args.validation_ratio < 0.5:
        raise SystemExit("validation-ratio must be in [0, 0.5).")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    sequences = read_sequences(args.data)
    random.shuffle(sequences)
    if len(sequences) >= 2 and args.validation_ratio:
        count = max(1, round(len(sequences) * args.validation_ratio))
        validation_sequences, training_sequences = sequences[:count], sequences[count:]
    else:
        training_sequences, validation_sequences = sequences, []
        print("WARNING: at least two independent MIDI sequences are needed for held-out validation.")

    vocabulary = SPECIAL + sorted({token for sequence in sequences for token in sequence if token not in SPECIAL})
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    training_data = EventDataset(training_sequences, token_to_id, args.block_size)
    validation_data = EventDataset(validation_sequences, token_to_id, args.block_size)
    if not len(training_data):
        raise SystemExit("Not enough training examples.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=args.batch_size) if len(validation_data) else None
    config = {"vocab_size": len(vocabulary), "block_size": args.block_size, "width": args.width,
              "layers": args.layers, "heads": args.heads, "dropout": 0.1}
    model = MusicTransformer(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    args.out.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    print(f"Device: {device} | sequences: {len(sequences)} | train: {len(training_sequences)} | validation: {len(validation_sequences)} | vocab: {len(vocabulary)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            batches += 1
        train_loss = total / max(1, batches)
        val_loss = mean_loss(model, val_loader, criterion, device) if val_loader else None
        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}" if val_loss is not None else f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss=n/a")
        checkpoint = {"model_state": model.state_dict(), "model_config": config, "epoch": epoch,
                      "train_loss": train_loss, "validation_loss": val_loss, "optimizer_state": optimizer.state_dict()}
        torch.save(checkpoint, args.out / "latest.pt")
        score = val_loss if val_loss is not None else train_loss
        if score < best_score:
            best_score = score
            torch.save(checkpoint, args.out / "best.pt")

    (args.out / "vocab.json").write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"model": "EJey Music Transformer", "version": "0.2.0",
              "task": "conditional symbolic MIDI event next-token prediction",
              "model_config": config, "epochs": args.epochs, "batch_size": args.batch_size,
              "learning_rate": args.learning_rate, "seed": args.seed, "device": str(device),
              "source_sequences": len(sequences), "training_sequences": len(training_sequences),
              "validation_sequences": len(validation_sequences), "training_examples": len(training_data),
              "best_validation_or_training_loss": best_score}
    (args.out / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training complete. Best checkpoint: {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
