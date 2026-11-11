#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

REPO_URL="https://github.com/domingosferrazfonseca283-blip/ejey-nexus-music.git"
WORKDIR="$HOME/ejey-nexus-music"
ZIP_PATH="$HOME/storage/downloads/EJey-Nexus-PC-Training.zip"

pkg update -y
pkg install -y git zip

if [ -d "$WORKDIR/.git" ]; then
  git -C "$WORKDIR" pull --ff-only
else
  git clone "$REPO_URL" "$WORKDIR"
fi

mkdir -p "$HOME/storage/downloads"
cd "$WORKDIR"
zip -r "$ZIP_PATH" backend training README.md \
  -x "backend/.env" "training/.venv/*" "training/data/raw/*" \
     "training/data/processed/*" "training/runs/*" "training/*.pt" "training/*.pth"

echo
echo "ZIP criado em: $ZIP_PATH"
echo "Copia este ficheiro para o PC. Os MIDI de treino devem ser adicionados no PC em training/data/raw/."
