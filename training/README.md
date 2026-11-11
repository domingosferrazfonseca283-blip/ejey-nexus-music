# EJey Nexus Music — PC training kit

## EJey Music Transformer v0.3

This package contains the first trainable EJey component: a conditional Transformer that learns symbolic note-event sequences from MIDI files. It generates MIDI drafts, not finished WAV/MP3 audio. It does not generate lyrics, vocals, album artwork or mastered audio; those remain separate specialist-model integrations on the roadmap.

## Requirements

- Windows, Linux or macOS
- Python 3.10 or newer
- 8 GB RAM is a practical starting point for small experiments
- NVIDIA GPU optional; CPU training is supported but slower

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Linux/macOS: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA acceleration, install the PyTorch build matching the PC's CUDA version using the official PyTorch installation selector, then install the remaining requirements.

## 1. Add and audit MIDI data

Copy only MIDI files you created, public-domain MIDI, or datasets whose licenses explicitly permit machine-learning training into `data/raw/`. Keep the license text and source records separately.

```bash
python audit_dataset.py --input data/raw --output data/processed/dataset_audit.json --license-label "public-domain"
```

The audit reports file hashes, duplicate files, tracks, note counts, durations and parsing errors. The license label is a user-supplied record; the script cannot verify rights.

## 2. Prepare labelled sequences

Each invocation applies the same labels to all MIDI files in that folder. For mixed genres or moods, use separate folders/output files, then combine the JSONL rows into one dataset.

```bash
python prepare_dataset.py --input data/raw --output data/processed/sequences.jsonl --genre "Jazz" --mood "Reflective" --tempo "Slow" --license-label "public-domain"
```

Use consistent spelling and capitalization. Generation conditions must exist in the training vocabulary.

## 3. Train

```bash
python train.py --data data/processed/sequences.jsonl --out runs/ejey-music-v0.3 --epochs 30
```

The trainer writes `latest.pt` every epoch, keeps the best checkpoint, writes `vocab.json` before training, and records `training_report.json`. It automatically selects CUDA when available, otherwise CPU.

Useful options:

- `--batch-size 8` to reduce memory use.
- `--block-size 128` for context length.
- `--width 192 --layers 4 --heads 4` for the starter network.
- `--validation-ratio 0.1` to reserve sequences for validation.
- `--seed 42` for repeatable initialization and split.

### Resume after interruption

Use the same dataset, output folder, model dimensions and seed. Increase the total target epoch count:

```bash
python train.py --data data/processed/sequences.jsonl --out runs/ejey-music-v0.3 --epochs 60 --resume runs/ejey-music-v0.3/latest.pt
```

Resume checks vocabulary and model dimensions. If the dataset changes, start a new run rather than silently mixing incompatible vocabularies.

## 4. Generate a MIDI draft

```bash
python generate.py --checkpoint runs/ejey-music-v0.3/best.pt --vocab runs/ejey-music-v0.3/vocab.json --output runs/generated/ejey-draft.mid --genre "Jazz" --mood "Reflective" --tempo "Slow" --max-tokens 256
```

Optional controls: `--temperature 0.9`, `--top-k 20`, `--program 0`, `--seed 1234`.

Open the MIDI in a DAW to listen, edit and arrange it. MIDI is symbolic note data, not rendered audio.

## Dataset quality and evaluation

- Prefer many independent, varied compositions over repeated copies.
- Keep validation compositions separate from training sources where possible.
- Inspect the audit report and remove corrupt files and duplicates.
- Monitor training and validation loss; lower loss alone does not guarantee musical quality.
- Listen to multiple generated samples and assess rhythm, pitch range, repetition, structure and musical coherence.

## Package contents and portability

The PC ZIP contains source code, requirements and instructions. It deliberately excludes raw MIDI, processed datasets, virtual environments, secrets and model checkpoints. Copy your licensed MIDI data to `training/data/raw/` on the PC. Keep training outputs under `training/runs/` and back them up separately.

## Roadmap

1. Conditional symbolic MIDI starter (current).
2. Richer multi-track event representation, instrument identity, tempo maps and song sections.
3. Composition and arrangement controls.
4. Audio generation/rendering through a specialist audio model.
5. Separate lyrics, synthetic vocals, mixing/mastering and image-generation models.
6. Secure orchestration API connecting specialist models to the EJey app.

This is a research starter, not a production music model. Its output quality depends on dataset rights, diversity, representation, model size and available compute.
