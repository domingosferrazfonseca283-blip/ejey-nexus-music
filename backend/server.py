import os
import secrets
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

VERSION = "0.2.0"
TOKEN = os.getenv("EJEY_API_TOKEN", "")
if not TOKEN:
    raise RuntimeError("Define EJEY_API_TOKEN antes de iniciar a API.")

app = FastAPI(title="EJey Nexus Music Engine", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["Authorization", "Content-Type"])

class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=10, max_length=5000)
    genre: str = Field(default="Outro", max_length=60)
    mood: str = Field(default="Inspirador", max_length=60)
    tempo: str = Field(default="Moderado", max_length=40)
    vocals: str = Field(default="Com voz", max_length=40)

def authenticate(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Chave de acesso em falta.")
    candidate = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(candidate, TOKEN):
        raise HTTPException(status_code=403, detail="Chave de acesso inválida.")

@app.get("/health")
def health():
    return {"status":"online","api_version":VERSION,"engine_status":"awaiting_model",
            "message":"API ligada. O modelo de geração de áudio ainda não está instalado."}

@app.post("/v1/generate", dependencies=[Depends(authenticate)])
def generate(request: GenerationRequest):
    raise HTTPException(status_code=503, detail="A API recebeu o pedido, mas ainda não existe um modelo musical integrado. A geração de áudio será ativada na próxima etapa.")
