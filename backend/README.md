# EJey Nexus Music Engine API

API inicial para ligar o aplicativo ao computador.

## Instalar no PC
Python 3.11 ou superior:
```bash
cd backend
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

Define uma chave privada:
```bash
# Linux/macOS
export EJEY_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# PowerShell: $env:EJEY_API_TOKEN = "uma-chave-longa-e-aleatoria"
```

Inicia a API:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

No aplicativo, configura `http://IP-LOCAL-DO-PC:8000` e a mesma chave. Usa HTTP apenas para testes numa rede Wi-Fi privada. Para acesso externo, configura HTTPS.

## Endpoints
- `GET /health`: verifica a API.
- `POST /v1/generate`: recebe prompts e parâmetros; exige Bearer token.

## Limitação atual
O endpoint de geração devolve HTTP 503 intencionalmente até existir um modelo de áudio integrado. A API não afirma gerar áudio nesta versão.
