#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
pkg update -y
pkg install -y python
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if [ ! -f .env ]; then
  TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
  printf 'EJEY_API_TOKEN=%s\n' "$TOKEN" > .env
  chmod 600 .env
  echo "Chave criada em backend/.env. Mantém-na privada."
fi
set -a
. ./.env
set +a
echo "API a iniciar na porta 8000."
uvicorn server:app --host 0.0.0.0 --port 8000
