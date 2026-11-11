# EJey Nexus Music

Estúdio móvel para criação musical assistida por IA.

## Aplicativo
Interface responsiva, construção de prompts, biblioteca local, teste da API do PC e envio de pedidos para geração.

## Motor no PC
A API FastAPI está em `backend/`. Consulta `backend/README.md`. O aplicativo envia a chave de acesso no cabeçalho Bearer.

## Termux
O script `backend/start-termux.sh` prepara o Python, instala dependências, cria uma chave privada e inicia a API. Termux é o ambiente de desenvolvimento; não substitui o processo de compilação Android.

## APK
O workflow em `.github/workflows/android.yml` compila um APK de depuração no GitHub Actions. O APK só fica disponível após uma execução bem-sucedida.

## Estado
A comunicação e o contrato da API estão preparados. A geração real de áudio ainda não está implementada: `POST /v1/generate` responde HTTP 503 até integrar um modelo.

## Arquitetura futura
- Telemóvel: interface, prompts, biblioteca e controlo.
- PC: API, inferência e processamento de áudio.
- Modelo: integração futura com licença e dados adequados.

Utiliza apenas gravações, letras, vozes e conjuntos de dados próprios ou autorizados. Nunca publiques a chave de acesso.
