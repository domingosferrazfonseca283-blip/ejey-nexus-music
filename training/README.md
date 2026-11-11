# EJey Nexus Music — treino do primeiro modelo

## Objetivo desta fase

Treinar o **EJey Music Transformer**, o primeiro componente próprio da plataforma. Nesta etapa ele aprende padrões de composição simbólica representados por eventos MIDI e prevê o próximo evento musical.

Este não é ainda um gerador de áudio, não cria capas e não representa o núcleo multimodal completo. A arquitetura será ampliada com módulos especializados depois de validarmos este primeiro modelo.

## Requisitos

- PC Windows, Linux ou macOS
- Python 3.10+
- 8 GB de RAM recomendados para experiências pequenas
- GPU NVIDIA opcional; CPU funciona, mas o treino pode ser lento

## 1. Preparar ambiente

A partir desta pasta:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Linux/macOS
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para GPU NVIDIA, instala a versão de PyTorch compatível com o CUDA do computador seguindo o seletor oficial do PyTorch. Depois instala as restantes dependências sem substituir a versão CUDA instalada.

## 2. Preparar os dados

Coloca ficheiros MIDI que tens direito a utilizar em `data/raw/`. O conversor aceita `.mid` e `.midi`, incluindo subpastas.

Utiliza composições próprias, domínio público ou datasets cuja licença permita explicitamente treino de modelos. Regista a origem e a licença dos conjuntos utilizados. Não uses material protegido sem autorização.

## 3. Converter MIDI

```bash
python prepare_dataset.py --input data/raw --output data/processed/sequences.jsonl
```

## 4. Iniciar o primeiro treino

```bash
python train.py --data data/processed/sequences.jsonl --out runs/ejey-music-v1 --epochs 30
```

Parâmetros úteis:

- `--batch-size 8`: reduz utilização de memória.
- `--block-size 128`: contexto de eventos por amostra.
- `--epochs 30`: número de passagens pelo conjunto.
- `--validation-ratio 0.1`: reserva parte das sequências para validação.
- `--width 192 --layers 4 --heads 4`: tamanho inicial da rede.

Começa com um conjunto pequeno para confirmar que o pipeline funciona antes de aumentar os dados ou o modelo.

## 5. Gerar uma primeira amostra MIDI\n\nDepois do treino:\n\n```bash\npython generate.py --checkpoint runs/ejey-music-v1/best.pt --vocab runs/ejey-music-v1/vocab.json --output runs/generated/ejey-sample.mid --max-tokens 256\n```\n\nO script amostra eventos do modelo e exporta um ficheiro MIDI e um JSON de eventos. A renderização usa um instrumento de piano padrão; esta é uma amostra técnica para validar a cadeia de geração, não uma produção musical final.\n\n## 6. Artefactos produzidos

A pasta `runs/ejey-music-v1/` contém:

- `best.pt`: checkpoint com melhor perda de validação (ou treino se não houver validação).
- `latest.pt`: checkpoint da última época.
- `vocab.json`: vocabulário de eventos.
- `training_report.json`: configuração e métricas do treino.

Guarda os checkpoints fora do Git e faz cópias de segurança.

## 7. Como avaliar

A perda (*loss*) mede a capacidade de prever eventos no conjunto reservado. Uma perda menor é útil, mas não garante música interessante. A próxima etapa deverá implementar geração controlada de eventos, conversão de eventos para MIDI, escuta crítica e avaliação de diversidade, coerência rítmica e estrutura musical.

## Limites atuais

- O modelo aprende eventos simbólicos MIDI, não formas de onda de áudio.
- Ainda não produz WAV/MP3 nem está ligado à API da aplicação.
- Ainda não gera capas, letras ou voz.
- A qualidade depende da quantidade, diversidade e qualidade dos dados licenciados.
- Não publiques tokens, chaves, dados privados ou checkpoints no repositório.
