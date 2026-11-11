# EJey Nexus Music

Estúdio móvel para criação musical assistida por IA.

## Aplicativo
Interface responsiva, construção de prompts, biblioteca local, teste da API do PC e envio de pedidos para geração.

## Motor no PC
A API FastAPI está em `backend/`. Consulta `backend/README.md`. O aplicativo envia a chave de acesso no cabeçalho Bearer.

## Treino experimental no PC
A pasta `training/` contém um pipeline inicial para converter ficheiros MIDI autorizados em sequências simbólicas e treinar um pequeno Transformer de previsão de eventos MIDI. Isto é um protótipo simbólico, não um gerador de áudio. Consulta `training/README.md`.

## Preparar ZIP no Termux
Executa no Termux:
```bash
termux-setup-storage
pkg update -y
pkg install -y git
git clone https://github.com/domingosferrazfonseca283-blip/ejey-nexus-music.git
cd ejey-nexus-music
bash package-termux.sh
```
O ficheiro `EJey-Nexus-PC-Training.zip` será criado em `~/storage/downloads/`. O ZIP inclui API e treino, mas exclui chaves, datasets e checkpoints. Adiciona no PC apenas MIDI próprios, de domínio público ou com licença que permita treino.

## APK
O workflow em `.github/workflows/android.yml` compila um APK de depuração no GitHub Actions. O APK está disponível como artefacto após uma execução bem-sucedida.

## Estado e limitações
A comunicação e o contrato da API estão preparados. A geração real de áudio ainda não está implementada: `POST /v1/generate` responde HTTP 503 até integrar um modelo de áudio. O modelo de treino incluído aprende sequências MIDI simbólicas e ainda não é carregado pela API.

## Arquitetura
- Telemóvel: interface, prompts, biblioteca e controlo.
- PC: API, treino experimental, inferência e futuro processamento de áudio.
- Modelo: integração futura com licença e dados adequados.

Nunca publiques a chave de acesso. Usa apenas gravações, letras, vozes e conjuntos de dados próprios ou autorizados.
