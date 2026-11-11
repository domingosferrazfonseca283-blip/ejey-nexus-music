#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

REPO_URL="https://github.com/domingosferrazfonseca283-blip/ejey-nexus-music.git"
WORKDIR="$HOME/ejey-nexus-music"
DOWNLOADS="$HOME/storage/downloads"
ZIP_PATH="$DOWNLOADS/EJey-Nexus-PC-Training.zip"

pkg update -y
pkg install -y git zip

if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  git clone "$REPO_URL" "$WORKDIR"
fi

mkdir -p "$DOWNLOADS"
cd "$WORKDIR"
rm -f "$ZIP_PATH"
zip -r "$ZIP_PATH" training README.md   -x "training/.venv/*" "training/**/__pycache__/*" "training/**/*.pyc" \
     "training/data/raw/*" "training/data/processed/*" "training/runs/*" \
     "training/*.pt" "training/*.pth" "training/.DS_Store"

unzip -t "$ZIP_PATH" >/dev/null
echo
echo "ZIP validado: $ZIP_PATH"
echo "Tamanho:"
du -h "$ZIP_PATH"
echo "No PC, extrai o ZIP, instala training/requirements.txt e coloca MIDI licenciados em training/data/raw/."
