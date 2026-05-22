## Setup

COATI is required for JEPA training because the target regression embedding is a COATI SMILES latent.

Install project dependencies (including COATI from GitHub):

```bash
pip install -e .
```

If your cluster has a problematic shared temp dir, force pip to use a private tmp location:

```bash
mkdir -p "$HOME/tmp"
TMPDIR="$HOME/tmp" pip install -e .
```

## Generate COATI embeddings

Generate and save SMILES embeddings before training:

```bash
python scripts/gen_coati_embeddings.py
```

By default this writes `coati_embeddings.pt` next to `train_csv`. You can override paths and device:

```bash
python scripts/gen_coati_embeddings.py \
  train_csv=/path/train.csv \
  val_csv=/path/val.csv \
  test_csv=/path/test.csv \
  smiles_embeddings_path=/path/coati_embeddings.pt \
  embedding_device=cuda:0
```

## Train

Point the datamodule at your generated embedding file:

```bash
python main.py datamodule.smiles_embeddings_path=/path/coati_embeddings.pt
```
