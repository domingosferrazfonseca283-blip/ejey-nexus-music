# EJey Nexus Music — pacote inicial de treino

Este pacote cria um **modelo experimental de sequência musical simbólica a partir de MIDI**. Não é um modelo de áudio e não produz WAV/MP3. A ideia é aprender padrões de notas e durações; depois será necessário ligar um sintetizador MIDI ou integrar um modelo de áudio licenciado.

## Requisitos

- PC Windows, Linux ou macOS
- Python 3.10+
- 8 GB RAM recomendados para experiências pequenas
- GPU NVIDIA opcional; CPU funciona, mas pode ser lenta

## 1. Preparar o ambiente

Na pasta `training`:

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
# Linux/macOS
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Para PyTorch com GPU NVIDIA, instala primeiro a versão adequada ao teu CUDA seguindo o seletor oficial do PyTorch. Depois instala os restantes pacotes deste ficheiro, sem substituir a instalação CUDA do PyTorch.

## 2. Organizar dados

Coloca ficheiros MIDI que tens direito a usar em:

```
training/data/raw/
```

São aceites `.mid` e `.midi`, incluindo subpastas. Usa composições próprias, domínio público ou conjuntos de dados cuja licença permita treino de modelos. Mantém um registo da origem e licença de cada conjunto. Não incluas gravações ou obras protegidas sem autorização.

## 3. Converter MIDI para sequências

A partir da pasta `training`:

```bash
python prepare_dataset.py --input data/raw --output data/processed/sequences.jsonl
```

Cada linha do JSONL contém uma sequência de eventos simbólicos. O script ignora MIDI sem notas e informa quantos ficheiros foram processados.

## 4. Treinar

```bash
python train.py --data data/processed/sequences.jsonl --out runs/ejey-midi-v1 --epochs 20
```

O treino grava `model.pt`, `vocab.json` e `training_config.json` na pasta indicada. Começa com poucas épocas e um conjunto pequeno para validar o processo.

## 5. Limitações e próximos passos

- É um protótipo de modelagem de eventos MIDI, não geração de áudio.
- O resultado depende da qualidade, diversidade e licença dos MIDI usados.
- Ainda não existe endpoint que carregue este checkpoint na API.
- Próxima fase: inferência/generação de MIDI, renderização com sintetizador e integração controlada com `backend/server.py`.
- Não coloques chaves, tokens ou dados privados dentro do ZIP ou do repositório.
