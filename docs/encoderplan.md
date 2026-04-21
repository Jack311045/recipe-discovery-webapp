# PLAN.md — Embeddings Module Implementation Plan
# Project: Recipe Discovery Web App

## 1. Objective

Implement the embeddings pipeline for the existing repository structure of the Recipe Discovery Web App.

This module is responsible for:

1. loading processed recipe data from `data/processed/`
2. constructing canonical recipe text representations
3. encoding each recipe into a dense semantic vector using a pre-trained Transformer
4. saving the embedding matrix and row-to-recipe mappings under `data/artifacts/`
5. building a cosine-similarity retrieval index for nearest-neighbor search

This work is foundational for downstream modules already present in the repo:

- `src/recipe_discovery/retrieval/`
- `src/recipe_discovery/clustering/`
- `src/recipe_discovery/models/`
- `src/recipe_discovery/reduction/`
- `src/recipe_discovery/evaluation/`

The implementation must match the existing repository layout exactly and should not introduce a parallel or conflicting structure.

---

## 2. Repository-Specific Scope

### In scope
Implement or update logic in the following existing files:

- `src/recipe_discovery/settings.py`
- `src/recipe_discovery/data/load.py`
- `src/recipe_discovery/data/corpus.py`
- `src/recipe_discovery/data/schema.py`
- `src/recipe_discovery/embeddings/encoder.py`
- `src/recipe_discovery/embeddings/store.py`
- `src/recipe_discovery/embeddings/index.py`
- `src/recipe_discovery/utils/io.py`
- `src/recipe_discovery/utils/validation.py`
- `scripts/build_embeddings.py`
- `scripts/build_index.py`

### Out of scope
Do not implement or modify the business logic of:

- `app/`
- `src/recipe_discovery/retrieval/service.py`
- `src/recipe_discovery/clustering/`
- `src/recipe_discovery/models/`
- `src/recipe_discovery/reduction/`
- `scripts/train_kmeans.py`
- `scripts/train_regression.py`
- `scripts/train_autoencoder.py`
- `scripts/fit_pca.py`
- `scripts/train_classifier.py`
- `scripts/evaluate_all.py`

Only add lightweight compatibility hooks if absolutely necessary.

---

## 3. Existing Repo Structure Constraints

This repository already uses the following structure:

- configs live in `configs/`
- processed input data lives in `data/processed/`
- generated embeddings and vector indices must live in `data/artifacts/`
- reusable source code must live under `src/recipe_discovery/`
- runnable entrypoints must live under `scripts/`

Do not create a new `src/embeddings/` directory.
Do not create a new top-level `embeddings/` package.
Do not duplicate functionality into a second path.

All implementation must use the existing package path:

- `src/recipe_discovery/embeddings/`

and related helpers under:

- `src/recipe_discovery/data/`
- `src/recipe_discovery/utils/`

---
## 4. Input Data Assumptions

Primary input file:

- `data/processed/Processed_data_updated2.csv`

The processed CSV already has a concrete schema. It includes at least the following important columns:

### Core identity / text-bearing columns
- `name`
- `recipe_id`
- `minutes`
- `n_steps`
- `steps`
- `description`
- `ingredients`
- `n_ingredients`

### One-hot tag/category columns
The dataset contains a very large block of binary indicator columns representing:
- time buckets
- cuisines
- dietary restrictions
- occasions
- ingredient families
- preparation methods
- meal types
- regional styles

Examples include:
- `15-minutes-or-less`
- `30-minutes-or-less`
- `american`
- `asian`
- `breakfast`
- `desserts`
- `easy`
- `gluten-free`
- `healthy`
- `high-protein`
- `indian`
- `italian`
- `kid-friendly`
- `low-carb`
- `low-fat`
- `main-dish`
- `quick-breads`
- `thai`
- `vegan`
- `vegetarian`
- `weeknight`

### Nutrition / target / interaction columns
- `calories`
- `total fat`
- `sugar`
- `sodium`
- `protein`
- `saturated fat`
- `carbohydrates`
- `rating`
- `num_ratings`
- `all_reviews`

### Important schema note
There appears to be an empty unnamed column after `n_ingredients`. The loader should detect and drop unnamed columns such as:
- `""`
- columns whose names match `^Unnamed`

The embeddings pipeline must not expect a single `tags` column. Instead, it must reconstruct a recipe's active tags from the one-hot binary columns.
## 4.1 Schema-Aware Feature Groups for Embeddings

The embeddings pipeline should organize columns into the following groups:

### A. Text-rich columns to serialize directly
These should be used directly in recipe text construction:
- `name`
- `description`
- `ingredients`
- `steps`

### B. Short numeric metadata to serialize as compact text
These may be included in the serialized text in concise human-readable form:
- `minutes`
- `n_steps`
- `n_ingredients`

### C. One-hot columns to convert back into tag text
All binary indicator columns should be scanned row-wise.
If a one-hot column has value `1`, its column name should be included in the reconstructed tag list.

These columns should be treated as semantic tags, not numeric continuous features.

### D. Numeric nutrition / popularity columns
These columns may optionally be appended as compact metadata text:
- `calories`
- `total fat`
- `sugar`
- `sodium`
- `protein`
- `saturated fat`
- `carbohydrates`
- `rating`
- `num_ratings`

### E. Large free-text review column
- `all_reviews`

This field may be extremely long. For the first implementation, do not dump the full raw review column into the serialized recipe text by default.
Preferred first-pass options:
1. omit `all_reviews` entirely, or
2. include only a short truncated preview, or
3. include it only behind a config flag

Checkpoint priority is a stable and reproducible pipeline, not maximal text length.


## 5. Functional Goal

The embeddings pipeline must support the following end-to-end workflow:

### Step 1
Load the processed CSV from `data/processed/Processed_data_updated2.csv`

### Step 2
Convert each recipe row into one canonical text string in:

- `src/recipe_discovery/data/corpus.py`

### Step 3
Load a pre-trained embedding model in:

- `src/recipe_discovery/embeddings/encoder.py`

### Step 4
Encode all recipe texts in batches into a dense matrix of shape:

- `(n_recipes, d_embedding)`

### Step 5
Save the embeddings and metadata mapping in:

- `data/artifacts/`

using utilities in:

- `src/recipe_discovery/embeddings/store.py`

### Step 6
Build a cosine-based retrieval index in:

- `src/recipe_discovery/embeddings/index.py`

### Step 7
Save the index under:

- `data/artifacts/`

so it can be loaded later by the retrieval layer

---

## 6. File-by-File Responsibilities

## 6.1 `src/recipe_discovery/settings.py`

This file should provide project-wide path resolution and config loading helpers.

Expected responsibilities:

- resolve repository root robustly using `pathlib.Path`
- expose canonical paths for:
  - `data/processed/`
  - `data/artifacts/`
  - `configs/base.yaml`
  - `configs/embeddings.yaml`
- optionally load YAML config for embeddings settings

Do not hardcode machine-specific absolute Windows paths.

---
## 6.2 `src/recipe_discovery/data/schema.py`

This file should define repo-specific schema groups for the processed recipe dataset.

Expected constants / helpers:

- `TEXT_COLUMNS = ["name", "description", "ingredients", "steps"]`
- `ID_COLUMN = "recipe_id"`
- `SHORT_METADATA_COLUMNS = ["minutes", "n_steps", "n_ingredients"]`
- `NUTRITION_COLUMNS = ["calories", "total fat", "sugar", "sodium", "protein", "saturated fat", "carbohydrates"]`
- `OUTCOME_COLUMNS = ["rating", "num_ratings"]`
- `LONG_TEXT_COLUMNS = ["all_reviews"]`

Also include helper logic such as:
- `is_unnamed_column(col_name) -> bool`
- `get_one_hot_tag_columns(df) -> list[str]`

### One-hot tag detection rule
A practical first-pass rule is:

A column is considered a tag column if it is:
- not in the known direct text / id / metadata / nutrition / outcome columns
- not unnamed
- and its non-null values are binary-like (`0/1`, `True/False`)

This logic should live here or in a closely related helper.

---

## 6.3 `src/recipe_discovery/data/load.py`

This file should contain reusable dataset loading utilities.

Expected responsibilities:

- load processed CSV from disk
- validate file existence
- optionally validate non-empty dataframe
- optionally provide a `load_processed_recipes()` helper

This module should not contain embedding-model logic.

---
## 6.4 `src/recipe_discovery/data/corpus.py`

This file is the single source of truth for converting one processed recipe row into one canonical embedding text.

Expected responsibilities:

- clean string fields
- drop or ignore unnamed columns
- detect one-hot tag columns automatically
- reconstruct active tags from binary columns
- serialize one recipe row into one deterministic text block
- serialize an entire dataframe into a sequence of recipe texts

### Required serialization logic

For each row:

1. Read direct text columns:
   - `name`
   - `description`
   - `ingredients`
   - `steps`

2. Read compact metadata:
   - `minutes`
   - `n_steps`
   - `n_ingredients`

3. Reconstruct tag text from all one-hot columns with value `1`

4. Optionally append nutrition summary:
   - `calories`
   - `protein`
   - `sodium`
   - etc.

5. Do not include `recipe_id` in the semantic text except optionally for debugging artifacts

6. Do not include the full `all_reviews` column by default in the first pass

### Preferred output format

Title: {name}
Description: {description}
Ingredients: {ingredients}
Steps: {steps}
Time: {minutes} minutes
Step Count: {n_steps}
Ingredient Count: {n_ingredients}
Tags: {comma-separated active one-hot labels}
Nutrition: calories={...}, protein={...}, sodium={...}

Requirements:
- skip missing values cleanly
- avoid literal `nan`
- keep formatting deterministic
- make the output readable during debugging
- keep one-hot tag extraction configurable and reusable
---

## 6.5 `src/recipe_discovery/embeddings/encoder.py`

This file should implement the text embedding wrapper.

Expected responsibilities:

- load the configured pre-trained sentence embedding model
- expose a reusable encoder class or functions
- support batch encoding
- optionally L2-normalize embeddings if desired for cosine retrieval
- return embeddings as NumPy arrays or Torch tensors, but keep saved artifact format consistent

Preferred default model:

- `sentence-transformers/all-MiniLM-L6-v2`

Requirements:

- isolate model loading here
- do not place model-loading code directly in the scripts if avoidable
- support configurable batch size
- produce reproducible, documented output shape

---

## 6.6 `src/recipe_discovery/embeddings/store.py`

This file should handle artifact persistence.

Expected responsibilities:

- ensure `data/artifacts/` exists
- save embedding matrix
- save row-to-recipe ID mapping
- save optional debug texts or metadata
- load previously saved embeddings if needed

Recommended saved artifact names:

- `data/artifacts/recipe_embeddings.npy`
- `data/artifacts/recipe_ids.csv`
- `data/artifacts/recipe_texts.csv`
- `data/artifacts/embedding_metadata.json`

Minimum required artifacts:

- embeddings matrix
- row-to-recipe mapping

---

## 6.7 `src/recipe_discovery/embeddings/index.py`

This file should build and persist the retrieval index.

Expected responsibilities:

- read embeddings
- apply normalization if needed for cosine search
- construct a nearest-neighbor index
- save index artifact
- load index artifact later for reuse

Preferred initial implementation:

- `sklearn.neighbors.NearestNeighbors(metric="cosine")`

This is preferred for checkpoint simplicity and portability.

If FAISS is already part of the project later, keep the code structured so the backend can be swapped, but do not overcomplicate the first pass.

Recommended saved index path:

- `data/artifacts/recipe_index.joblib`

---

## 6.8 `src/recipe_discovery/utils/io.py`

Use this only for shared file I/O helpers if needed.

Possible responsibilities:

- create directory safely
- read/write JSON
- read/write YAML
- small reusable serialization helpers

Do not move all embeddings logic here.
Keep embeddings-specific persistence in `src/recipe_discovery/embeddings/store.py`.

---

## 6.9 `src/recipe_discovery/utils/validation.py`

Use this only for generic validation helpers if useful.

Possible responsibilities:

- validate required file exists
- validate dataframe is non-empty
- validate required columns are resolvable
- validate embedding matrix shape

Keep validation helpers small and reusable.

---

## 6.10 `scripts/build_embeddings.py`

This script is the offline entrypoint for generating recipe embeddings.

It must:

1. resolve config and paths
2. load `data/processed/Processed_data_updated2.csv`
3. build recipe texts through `src/recipe_discovery/data/corpus.py`
4. load the embedding model via `src/recipe_discovery/embeddings/encoder.py`
5. encode all texts in batches
6. save artifacts using `src/recipe_discovery/embeddings/store.py`
7. print concise summary logs

Expected script behavior:

```bash
python scripts/build_embeddings.py

Optional supported arguments:

--input-path
--output-dir
--model-name
--batch-size
--overwrite

The script must be runnable from the repository root.

6.11 scripts/build_index.py

This script is the offline entrypoint for building the vector retrieval index.

It must:

load saved embeddings from data/artifacts/
build the cosine nearest-neighbor index
save the index artifact
print concise summary logs

Expected script behavior:

python scripts/build_index.py

Optional supported arguments:

--embedding-path
--output-path
--overwrite

The script must also be runnable from the repository root.

7. Config Integration

Use the existing config directory:

configs/base.yaml
configs/embeddings.yaml
configs/base.yaml

Can contain shared defaults such as:

project paths
random seed
logging level
configs/embeddings.yaml

Should contain embeddings-specific settings such as:

model name
batch size
normalization flag
artifact filenames
overwrite behavior

Do not create a second config system outside configs/.

If config loading is not fully implemented yet, implement only the minimum clean path necessary and keep the structure compatible with future expansion.

8. Artifact Contract

The embeddings module should produce artifacts that downstream modules can rely on.

Minimum required outputs under data/artifacts/:

recipe_embeddings.npy
recipe_ids.csv
recipe_index.joblib

Optional but strongly recommended:

recipe_texts.csv
embedding_metadata.json
Required contract
each row in recipe_embeddings.npy must correspond exactly to the same row/order in recipe_ids.csv
if recipe_texts.csv is saved, it must align row-for-row as well
shapes and counts must be validated before saving when practical
9. Robustness Requirements

The implementation must:

work on Windows local development
use pathlib.Path
avoid hardcoded absolute paths
fail with clear error messages if input CSV is missing
handle missing optional columns safely
handle empty strings and missing values safely
avoid regenerating artifacts unless overwrite is explicitly requested
keep logs concise and readable

The implementation should prefer stability and clarity over premature optimization.

10. Modeling Choice for First Pass

For the initial implementation, prioritize a lightweight and stable embedding setup.

Default embedding model
sentence-transformers/all-MiniLM-L6-v2
Default index backend
sklearn.neighbors.NearestNeighbors(metric="cosine")

The goal of this phase is not maximum retrieval quality yet.
The goal is a clean, reproducible, working embeddings pipeline that integrates with the existing repo.

## 10.1 Script-specific expected behavior for this dataset

### `scripts/build_embeddings.py`
The script must:
- load `data/processed/Processed_data_updated2.csv`
- drop unnamed columns
- identify one-hot tag columns
- reconstruct active tag text row-by-row
- serialize each recipe into one canonical text
- encode the text into dense vectors
- save:
  - `data/artifacts/recipe_embeddings.npy`
  - `data/artifacts/recipe_ids.csv`
  - optionally `data/artifacts/recipe_texts.csv`

### `scripts/build_index.py`
The script must:
- load `recipe_embeddings.npy`
- build cosine nearest-neighbor index
- save:
  - `data/artifacts/recipe_index.joblib`

11. Suggested Implementation Order

Implement in this order:

path resolution and config access in src/recipe_discovery/settings.py
processed CSV loading in src/recipe_discovery/data/load.py
column resolution helpers in src/recipe_discovery/data/schema.py
recipe text serialization in src/recipe_discovery/data/corpus.py
embedding model wrapper in src/recipe_discovery/embeddings/encoder.py
artifact save/load utilities in src/recipe_discovery/embeddings/store.py
scripts/build_embeddings.py
index construction in src/recipe_discovery/embeddings/index.py
scripts/build_index.py
minimal smoke validation on a small subset
full run on the processed dataset
12. Acceptance Criteria

This task is complete only if all of the following are true:

python scripts/build_embeddings.py runs successfully from repo root
data/processed/Processed_data_updated2.csv is loaded correctly
canonical recipe texts are generated without crashing on missing optional fields
embeddings are generated for the full processed dataset
embeddings and row mapping are saved under data/artifacts/
python scripts/build_index.py runs successfully from repo root
a cosine retrieval index is saved under data/artifacts/
code uses the existing repo structure exactly
no parallel duplicate package structure is introduced
code contains docstrings and clear function boundaries
downstream retrieval code can later load artifacts without changing artifact locations
13. Non-Goals and Constraints

Do not:

create a new package path outside src/recipe_discovery/
move unrelated files
implement Streamlit pages in this task
implement clustering or regression in this task
over-engineer the first pass with unnecessary abstractions
assume one fixed schema if the processed CSV may vary slightly

Do:

preserve the current repo architecture
write production-quality starter code
keep each file focused on one responsibility
make the output artifacts predictable and reusable
14. Deliverable Style

The final implementation should look like professional project infrastructure, not notebook-style experimental code.

It should have:

small focused functions
readable docstrings
stable path handling
clear script entrypoints
deterministic artifact names
minimal but useful logging
compatibility with future retrieval integration
15. Important Repo-Specific Reminder

This project already defines the embeddings responsibility as:

src/recipe_discovery/embeddings/
scripts/build_embeddings.py
scripts/build_index.py

Do not re-interpret this task as a generic standalone script.
The implementation must fit the exact repository layout listed above and integrate cleanly with the rest of the project.