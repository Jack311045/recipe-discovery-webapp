# Embeddings Pipeline — Runbook

## Prerequisites

1. **Python 3.11+** installed
2. All dependencies installed: `pip install -r requirements.txt`
3. The processed CSV exists at `data/processed/Processed_data_updated2.csv`
4. Run all commands from the **repository root** directory

---

## Step 1: Build Embeddings

```bash
python scripts/build_embeddings.py
```

**What it does:**
1. Loads `data/processed/Processed_data_updated2.csv` (226,657 rows, 504 columns)
2. Drops unnamed columns (`Unnamed: 8`)
3. Detects 485 one-hot tag columns automatically
4. Builds a canonical text representation for each recipe
5. Loads the `all-MiniLM-L6-v2` sentence-transformer model
6. Encodes all 226,657 recipes into 384-dimensional vectors
7. Saves four artifacts to `data/artifacts/`

**Expected output files:**

| File | Size (approx.) |
|---|---|
| `data/artifacts/recipe_embeddings.npy` | ~330 MB |
| `data/artifacts/recipe_ids.csv` | ~2 MB |
| `data/artifacts/recipe_texts.csv` | ~200 MB |
| `data/artifacts/embedding_metadata.json` | <1 KB |

**Expected runtime:** 15–45 minutes on CPU, depending on hardware.

### Optional flags

```bash
# Custom input path
python scripts/build_embeddings.py --input-path path/to/other.csv

# Larger batch size (faster if you have enough RAM)
python scripts/build_embeddings.py --batch-size 64

# Overwrite existing artifacts
python scripts/build_embeddings.py --overwrite

# Different model
python scripts/build_embeddings.py --model-name sentence-transformers/all-mpnet-base-v2
```

---

## Step 2: Build Retrieval Index

```bash
python scripts/build_index.py
```

**What it does:**
1. Loads `data/artifacts/recipe_embeddings.npy`
2. Fits a `sklearn.neighbors.NearestNeighbors(metric="cosine")` index
3. Saves the index as `data/artifacts/recipe_index.joblib`

**Expected runtime:** A few seconds.

### Optional flags

```bash
# Custom embedding path
python scripts/build_index.py --embedding-path data/artifacts/recipe_embeddings.npy

# Custom output path
python scripts/build_index.py --output-path data/artifacts/my_index.joblib

# Overwrite existing index
python scripts/build_index.py --overwrite
```

---

## Step 3: Verify Artifacts

Quick check to confirm everything is correct:

```python
import numpy as np, pandas as pd, joblib

emb = np.load("data/artifacts/recipe_embeddings.npy")
ids = pd.read_csv("data/artifacts/recipe_ids.csv")
idx = joblib.load("data/artifacts/recipe_index.joblib")

print("Embeddings:", emb.shape)      # (226657, 384)
print("IDs:", len(ids))              # 226657
print("Index samples:", idx.n_samples_fit_)  # 226657

# Row alignment
assert emb.shape[0] == len(ids)

# Quick query
dists, inds = idx.kneighbors(emb[:1], n_neighbors=5)
print("Nearest 5:", ids.iloc[inds[0]].values)
```

---

## Common Failure Cases

### 1. `FileNotFoundError: Processed CSV not found`

**Cause:** The CSV is not at the expected path.

**Fix:** Either place it at `data/processed/Processed_data_updated2.csv` or pass `--input-path`:

```bash
python scripts/build_embeddings.py --input-path data/processed/YOUR_FILE.csv
```

### 2. `RuntimeError: sentence-transformers is not installed`

**Fix:**

```bash
pip install sentence-transformers
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### 3. `ModuleNotFoundError: No module named 'recipe_discovery'`

**Cause:** The `src/` directory is not on the Python path.

**Fix:** Run from the repo root. The scripts automatically add `src/` to `sys.path`. If running from a different directory, set:

```bash
set PYTHONPATH=src    # Windows
export PYTHONPATH=src # Linux/Mac
```

### 4. `WARNING: Embeddings already exist. Use --overwrite`

**Cause:** Artifacts from a previous run exist.

**Fix:** Add `--overwrite` to regenerate:

```bash
python scripts/build_embeddings.py --overwrite
python scripts/build_index.py --overwrite
```

### 5. Out of Memory (OOM) during encoding

**Cause:** The full dataset (226K recipes) may strain systems with <8 GB RAM.

**Fix:** Reduce batch size:

```bash
python scripts/build_embeddings.py --batch-size 8
```

### 6. Schema mismatch / unexpected columns

**Cause:** A different processed CSV with a different column layout.

**Debug:** Inspect the CSV:

```python
import pandas as pd
df = pd.read_csv("data/processed/Processed_data_updated2.csv", nrows=5)
print(list(df.columns))
```

The pipeline will skip missing optional columns gracefully, but requires at minimum `recipe_id` and `name`.

---

## Debugging Tips

### Inspect a single recipe's canonical text

```python
import sys; sys.path.insert(0, "src")
from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.data.corpus import build_corpus

df = load_processed_recipes()
df = build_corpus(df.head(1))
print(df["recipe_text"].iloc[0])
```

### Check one-hot tag columns

```python
from recipe_discovery.data.schema import get_one_hot_tag_columns
tags = get_one_hot_tag_columns(df)
print(f"{len(tags)} tag columns detected")
print(tags[:20])
```

### Inspect saved metadata

```python
import json
meta = json.load(open("data/artifacts/embedding_metadata.json"))
print(json.dumps(meta, indent=2))
```

### Query the index manually

```python
import sys; sys.path.insert(0, "src")
from recipe_discovery.embeddings.encoder import EmbeddingConfig, RecipeEncoder
from recipe_discovery.embeddings.index import load_index
from recipe_discovery.embeddings.store import load_recipe_ids

encoder = RecipeEncoder(EmbeddingConfig())
encoder.load()

query_vec = encoder.encode(["spicy thai chicken stir fry"])
index = load_index()
ids = load_recipe_ids()

dists, inds = index.kneighbors(query_vec, n_neighbors=5)
for i, (idx_val, dist) in enumerate(zip(inds[0], dists[0])):
    print(f"{i+1}. recipe_id={ids.iloc[idx_val]}, distance={dist:.4f}")
```
