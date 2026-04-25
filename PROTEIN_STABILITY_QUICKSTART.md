# Protein Stability Dataset Quickstart

## Fastest dataset for your 30-40 minute window

Use **Hugging Face `RosettaCommons/MegaScale` with `dataset3_single`**.

Why this is the best fit right now:

- it is already hosted in a format that `datasets.load_dataset(...)` can pull directly
- it is focused on **single-point mutations**
- it includes **`ddG_ML`** for regression
- it includes **`Stabilizing_mut`** for binary classification
- it is much faster to wire into a ProtBERT pipeline than scraping raw mutation tables

The tradeoff is that MegaScale mostly contains short domains rather than full-length natural proteins. For a first prototype, that is usually acceptable.

## Recommended label setup

- **Regression**: use `ddg`
- **Binary classification**: use `label`

The standardized CSVs created by `data/protein_stability_dataset.py` contain:

- `protein_id`
- `wildtype_sequence`
- `mutation`
- `mutant_sequence`
- `ddg`
- `label`
- `split`

## Install the missing packages

```powershell
py -3 -m pip install -r requirements.txt
```

## Preview the dataset without saving files

```powershell
py -3 data\protein_stability_dataset.py --limit 1000 --preview-only
```

## Save train/val/test CSVs for a quick prototype

```powershell
py -3 data\protein_stability_dataset.py --limit 20000 --output-dir data\protein_stability
```

That gives you a smaller subset per split so you can start training a frozen ProtBERT head quickly.

## Minimal loading snippet for training

```python
import pandas as pd

train_df = pd.read_csv("data/protein_stability/megascale_train.csv")
val_df = pd.read_csv("data/protein_stability/megascale_val.csv")
test_df = pd.read_csv("data/protein_stability/megascale_test.csv")

print(train_df[["wildtype_sequence", "mutation", "mutant_sequence", "ddg", "label"]].head())
```

## If you want a more literature-style curated source later

Use **FireProtDB 2.0** as a second-stage dataset source:

- download page: https://loschmidt.chemi.muni.cz/fireprotdb/download/
- paper/archive: https://zenodo.org/records/18256279

FireProtDB is excellent for curated experimental mutation data, but it takes more cleanup than MegaScale for a same-day prototype.
