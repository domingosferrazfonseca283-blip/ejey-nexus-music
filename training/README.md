# EJey Nexus Music — AI model training

## Current milestone: EJey Music Transformer v0.2

The first trainable component is a **conditional symbolic-music model**. It learns event sequences from MIDI and can generate new MIDI drafts conditioned on genre, mood and tempo labels represented in its training vocabulary.

This is not yet a complete music-production AI. It does not directly generate WAV/MP3, lyrics, vocals, album artwork or mastered audio. Those require separate specialist models/services and an orchestration layer. The app's album-cover canvas remains a local procedural graphics tool until an image model is integrated.

## Training environment

Use a Windows, Linux or macOS PC. Python 3.10+ is recommended; 8 GB RAM is a practical starting point. A compatible NVIDIA GPU can accelerate training, but CPU mode is supported for small experiments.

```bash
python -m venv .venv
```

Windows PowerShell: `.venv\\Scripts\\Activate.ps1`

Linux/macOS: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For NVIDIA GPU support, install the PyTorch build matching the computer's CUDA environment using the official PyTorch selector, then install the remaining requirements.

## Dataset and rights

Put MIDI files in `data/raw/`. Use only music you created, public-domain material, or datasets whose licenses explicitly permit machine-learning training. Keep a record of each source and its license. Do not upload private material or copyrighted recordings/MIDI without permission.

## 1. Prepare labelled MIDI sequences

The metadata flags apply to every MIDI file in the input folder. For a mixed dataset, prepare separate labelled subsets and combine their JSONL output files afterward.

```bash
python prepare_dataset.py --input data/raw --output data/processed/sequences.jsonl --genre "Jazz" --mood "Reflective" --tempo "Slow" --license-label "public-domain"
```

Allowed labels are free text, but generation must use labels represented in the training vocabulary. Use consistent spelling and capitalization.

## 2. Train

```bash
python train.py --data data/processed/sequences.jsonl --out runs/ejey-music-v0.2 --epochs 30
```

Useful options:

- `--batch-size 8` to reduce memory use.
- `--block-size 128` for context length.
- `--width 192 --layers 4 --heads 4` for the starter network.
- `--validation-ratio 0.1` to reserve independent MIDI sequences for validation.

Start with a small experiment and inspect training and validation loss. A lower loss does not by itself mean better music. Keep validation files separate by composition/source where possible to reduce leakage.

## 3. Generate a MIDI draft

Use genre, mood and tempo labels that exist in the training vocabulary:

```bash
python generate.py --checkpoint runs/ejey-music-v0.2/best.pt --vocab runs/ejey-music-v0.2/vocab.json --output runs/generated/ejey-draft.mid --genre "Jazz" --mood "Reflective" --tempo "Slow" --max-tokens 256
```

Optional controls: `--temperature 0.9`, `--top-k 20`, `--program 0`, `--seed 1234`.

Open the MIDI in a DAW to listen, edit and arrange it. MIDI is symbolic note data; it is not a rendered audio file.

## Output and evaluation

The training run writes `best.pt`, `latest.pt`, `vocab.json` and `training_report.json`. Evaluate held-out loss, valid note ranges, rhythmic variation, pitch distribution, repetition and musical quality by listening to multiple samples.

## Model roadmap

1. Conditional symbolic MIDI (current starter).
2. Multi-track representation with instrument identity, tempo maps, bars and richer timing.
3. Arrangement and composition controls, including song sections and instrumentation.
4. Audio rendering / text-to-audio integration through a separate audio model.
5. Separate modules for lyrics, synthetic vocals, mixing/mastering and AI album artwork.
6. A secure orchestration API that connects specialist models to the EJey app.

Do not treat the starter Transformer as a production music model. Its quality is limited by dataset size, representation, labels and compute. Train on a PC; the phone app is the client, not the training machine.
