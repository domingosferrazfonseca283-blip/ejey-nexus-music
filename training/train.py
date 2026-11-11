#!/usr/bin/env python3
"""Train EJey's first symbolic-music Transformer from licensed MIDI event data."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SPECIAL = ["<PAD>", "<BOS>", "<EOS>"]


class EventDataset(Dataset):
    def __init__(self, sequences, token_to_id, block_size):
        self.examples = []
        for sequence in sequences:
            ids = [token_to_id["<BOS>"]]
            ids.extend(token_to_id[token] for token in sequence)
            ids.append(token_to_id["<EOS>"])
            for start in range(0, len(ids) - 1, block_size):
                part = ids[start:start + block_size + 1]
                if len(part) > 1:
                    inputs = part[:-1]
                    targets = part[1:]
                    pad = block_size - len(inputs)
                    self.examples.append((
                        inputs + [token_to_id["<PAD>"]] * pad,
                        targets + [-100] * pad
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
            raise ValueError("width tem de ser divisível por heads")
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.position_embedding = nn.Embedding(block_size, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, token_ids):
        batch_size, length = token_ids.shape
        if length > self.block_size:
            raise ValueError(f"sequência excede block_size={self.block_size}")
        positions = torch.arange(length, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = token_ids.eq(0)
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        return self.output(self.norm(hidden))


def read_sequences(path):
    sequences = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"Linha {line_number} ignorada: {error}")
                continue
            events = row.get("events") if isinstance(row, dict) else None
            if isinstance(events, list) and len(events) >= 2 and all(
                isinstance(event, str) for event in events
            ):
                sequences.append(events)
    if not sequences:
        raise SystemExit("Dataset vazio/inválido. Executa prepare_dataset.py primeiro.")
    return sequences


def mean_loss(model, loader, criterion, device):
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item()
            batches += 1
    return total / max(1, batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/processed/sequences.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("runs/ejey-music-v1"))
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

    if args.epochs < 1 or args.batch_size < 1 or args.block_size < 8:
        raise SystemExit("epochs/batch-size devem ser positivos e block-size >= 8.")
    if args.width < 32 or args.layers < 1 or args.heads < 1 or args.width % args.heads:
        raise SystemExit("Configuração inválida de width/layers/heads.")
    if not 0.0 <= args.validation_ratio < 0.5:
        raise SystemExit("validation-ratio deve estar entre 0 e 0.5.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sequences = read_sequences(args.data)
    random.shuffle(sequences)
    if len(sequences) >= 2 and args.validation_ratio > 0:
        validation_count = max(1, round(len(sequences) * args.validation_ratio))
        validation_sequences = sequences[:validation_count]
        training_sequences = sequences[validation_count:]
    else:
        training_sequences = sequences
        validation_sequences = []
        print("AVISO: são necessárias pelo menos 2 sequências para validação separada.")

    vocabulary = SPECIAL + sorted({
        token for sequence in sequences if sequence
        for token in sequence if token not in SPECIAL
    })
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    training_data = EventDataset(training_sequences, token_to_id, args.block_size)
    validation_data = EventDataset(validation_sequences, token_to_id, args.block_size)
    if not len(training_data):
        raise SystemExit("Não existem exemplos de treino suficientes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = (
        DataLoader(validation_data, batch_size=args.batch_size)
        if len(validation_data) else None
    )
    model_config = {
        "vocab_size": len(vocabulary),
        "block_size": args.block_size,
        "width": args.width,
        "layers": args.layers,
        "heads": args.heads,
        "dropout": 0.1,
    }
    model = MusicTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    args.out.mkdir(parents=True, exist_ok=True)

    print(
        f"Dispositivo: {device} | MIDI/sequências: {len(sequences)} | "
        f"treino: {len(training_sequences)} | validação: {len(validation_sequences)} | "
        f"exemplos: {len(training_data)} | vocabulário: {len(vocabulary)}"
    )

    best_validation = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
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
        validation_loss = (
            mean_loss(model, validation_loader, criterion, device)
            if validation_loader else None
        )
        scheduler.step()
        val_label = f"{validation_loss:.4f}" if validation_loss is not None else "n/a"
        print(
            f"Época {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_label}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "optimizer_state": optimizer.state_dict(),
        }
        torch.save(checkpoint, args.out / "latest.pt")
        score = validation_loss if validation_loss is not None else train_loss
        if score < best_validation:
            best_validation = score
            torch.save(checkpoint, args.out / "best.pt")

    (args.out / "vocab.json").write_text(
        json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "model": "EJey Music Transformer",
        "version": "0.1.0",
        "task": "symbolic MIDI event next-token prediction",
        "model_config": model_config,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "source_sequences": len(sequences),
        "training_sequences": len(training_sequences),
        "validation_sequences": len(validation_sequences),
        "training_examples": len(training_data),
        "best_validation_or_training_loss": best_validation,
    }
    (args.out / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Treino concluído. Melhor checkpoint: {args.out / 'best.pt'}")


if __name__ == "__main__":
    main()
