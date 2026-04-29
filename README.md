# Recipe Discovery Web App

A semantic recipe discovery application built on the Food.com Recipes and Interactions dataset.

## Project goal

Given a natural-language food request and optional dietary constraints, return the most relevant recipes, organize them into semantic clusters, and visualize recipe relationships in 2D.

## Course methods represented in this repository

- Dense text embeddings for recipes
- Cosine-similarity nearest-neighbor retrieval
- K-means clustering on recipe embeddings
- Regression model for recipe quality or rating prediction
- PCA and deep autoencoder for 2D visualization
- Optional classification module for recipe tag prediction
- Explicit train / validation / test support and k-fold hooks for evaluation

## What counts as a “working environment”

For the checkpoint, a working environment usually means:

1. A teammate or TA can clone the repo.
2. They can create a virtual environment.
3. They can install the dependencies from `requirements.txt`.
4. They can run the app entrypoint without import errors or missing-file crashes.
5. The repo already contains the module stubs described in the design document, even if the models are still placeholders.

That does **not** mean the final ML system has to be fully trained yet. It means the project is organized, runnable, and ready for implementation.

## Quick start

### 1) Create a virtual environment

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Add processed data + artifacts (required)
The app expects a processed CSV plus embedding artifacts. These are stored
outside of Git due to size.

#### 3a) Add processed CSV
Place the processed file at:
- `data/processed/Processed_data_updated2.csv`

If you already have a local `Processed_data_updated2.csv`, move it into
`data/processed/`.

#### 3b) Fetch embedding artifacts (Google Drive)
Download the shared artifacts into `data/artifacts/`:

```bash
pip install -r requirements.txt
python scripts/fetch_gdrive_artifacts.py --folder "https://drive.google.com/drive/folders/1bzKGQINcPabu0nIFnEqkJQwMlJgVRPLK?usp=sharing"
```

You can also set it once and reuse it:
```bash
export GDRIVE_ARTIFACTS_FOLDER="https://drive.google.com/drive/folders/1bzKGQINcPabu0nIFnEqkJQwMlJgVRPLK?usp=sharing"
python scripts/fetch_gdrive_artifacts.py
```

#### Optional: raw Kaggle data (only if you plan to rebuild)
If you want to rebuild the processed CSV, you will need the Kaggle raw files:
- `data/raw/RAW_recipes.csv`
- `data/raw/RAW_interactions.csv`

### 3b) Fetch embedding artifacts (Google Drive)
This repo keeps large embedding artifacts out of Git. To pull them from Google Drive:

1) Upload everything from `data/artifacts/` to a shared Google Drive folder.
2) Set the folder sharing to "Anyone with the link" (Viewer).
3) Copy the folder link or ID.
4) Download the artifacts locally:

```bash
pip install -r requirements.txt
python scripts/fetch_gdrive_artifacts.py --folder "https://drive.google.com/drive/folders/1bzKGQINcPabu0nIFnEqkJQwMlJgVRPLK?usp=sharing"
```

You can also set it once and reuse it:
```bash
export GDRIVE_ARTIFACTS_FOLDER="https://drive.google.com/drive/folders/1bzKGQINcPabu0nIFnEqkJQwMlJgVRPLK?usp=sharing"
python scripts/fetch_gdrive_artifacts.py
```

The current preprocessing script is minimal and does not persist outputs.
Rebuilding is not required for running the app.

### 4) Run the app
```bash
streamlit run app/streamlit_app.py
```

## Repository map

- `app/`: Streamlit user interface
- `configs/`: YAML config files
- `data/`: local-only raw, interim, processed, and artifact storage
- `docs/`: proposal and design docs
- `notebooks/`: exploration notebooks
- `scripts/`: pipeline and training scripts
- `src/recipe_discovery/`: main package
- `tests/`: smoke and unit tests

## Expected MVP workflow

1. Clean and merge recipe data
2. Build recipe text fields
3. Compute recipe embeddings
4. Run cosine retrieval for user queries
5. Fit k-means clusters for exploration
6. Fit regression ranking signal
7. Produce 2D projection with PCA or autoencoder
8. Surface everything in Streamlit

## Current status

This repository is a professional scaffold for the checkpoint:
- directory structure is in place
- imports are organized
- module stubs exist for all major methods
- app can run as a placeholder UI
