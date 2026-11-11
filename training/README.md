# EJey Nexus Music — treino do primeiro modelo musical

## Objetivo

Treinar e validar o **EJey Music Transformer v0.1**, o primeiro modelo próprio da plataforma. Ele aprende sequências simbólicas extraídas de MIDI e gera novas sequências exportáveis como MIDI.

Esta é a primeira etapa do motor musical, não o sistema multimodal completo. Geração de áudio WAV/MP3, capas por IA, letras e síntese vocal serão módulos separados, integrados depois através da API.

## Requisitos

- PC Windows, Linux ou macOS
- Python 3.10+
- 8 GB de RAM recomendados para experiências pequenas
- GPU NVIDIA opcional; CPU funciona, mas é mais lenta

## 1. Ambiente

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Instala as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para GPU NVIDIA, instala a versão de PyTorch compatível com o CUDA do computador, seguindo o seletor oficial do PyTorch. Depois instala as restantes dependências sem substituir essa versão.

## 2. Dados de treino

Coloca ficheiros MIDI que tens direito a utilizar em `data/raw/`. O conversor percorre subpastas e aceita `.mid` e `.midi`.

Usa composições próprias, domínio público ou conjuntos cuja licença permita explicitamente treino de modelos. Regista a origem, licença e condições de uso de cada conjunto. Não uses material protegido sem autorização.

## 3. Converter MIDI

```bash
python prepare_dataset.py --input data/raw --output data/processed/sequences.jsonl
```

O conversor representa notas com eventos de espera, altura, intensidade e duração. Nesta versão, as pistas MIDI são combinadas numa sequência e a saída inicial usa um instrumento.

## 4. Treinar

```bash
python train.py --data data/processed/sequences.jsonl --out runs/ejey-music-v1 --epochs 30
```

Parâmetros ajustáveis:

- `--batch-size 8`: reduz consumo de memória.
- `--block-size 128`: contexto de eventos.
- `--epochs 30`: número de épocas.
- `--validation-ratio 0.1`: fração reservada para validação.
- `--width 192 --layers 4 --heads 4`: dimensão inicial da rede.

Começa com uma experiência pequena. Aumenta o modelo apenas depois de verificar memória, tempo e métricas.

## 5. Gerar uma amostra MIDI

```bash
python generate.py --checkpoint runs/ejey-music-v1/best.pt --vocab runs/ejey-music-v1/vocab.json --output runs/generated/ejey-sample.mid --max-tokens 256
```

Opções:

- `--temperature 0.9`: controla a diversidade.
- `--top-k 20`: limita a amostragem aos tokens mais prováveis; 0 desativa o limite.
- `--program 0`: instrumento General MIDI; 0 é piano.
- `--seed 1234`: semente reproduzível.

O programa cria também um JSON com os eventos gerados. Abre o MIDI num DAW ou editor MIDI para avaliação.

## 6. Artefactos e avaliação

A pasta de saída contém `best.pt`, `latest.pt`, `vocab.json` e `training_report.json`.

Avalia perda de treino e validação, diversidade entre amostras, coerência rítmica, distribuição de alturas, duração e qualidade ao ouvir os MIDI. Uma perda baixa não garante música de qualidade.

## 7. Limites desta versão

- Aprende eventos MIDI simbólicos, não áudio bruto.
- Não produz WAV/MP3 nem está ligado ao endpoint de geração da aplicação.
- Não cria capas, letras ou vozes.
- A representação simplifica tempo, instrumentos e polifonia.
- A qualidade depende da quantidade, diversidade e licenciamento dos dados.
- Não publiques tokens, chaves, dados privados, MIDI de terceiros ou checkpoints no Git.

## Próxima evolução

Depois de validar treino e geração, a próxima versão deverá introduzir condicionamento por género, andamento, ambiente e estrutura; depois, representação multifaixa com instrumentos e compassos. A geração de áudio será um componente distinto, sem confundir MIDI simbólico com som final.
