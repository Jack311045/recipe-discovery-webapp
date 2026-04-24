"""Model Diagnostics — shows PCA explained variance, embedding info, and artifact status."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import json

import numpy as np
import streamlit as st

from recipe_discovery.settings import ARTIFACTS_DIR

st.set_page_config(page_title="Model Diagnostics", layout="wide")
st.title("🔬 Model Diagnostics")
st.caption("Verify that all artifacts are present and inspect model characteristics.")

# ── Artifact status ──────────────────────────────────────────────────────────
st.subheader("Artifact Status")

artifacts = {
    "Embeddings": ARTIFACTS_DIR / "recipe_embeddings.npy",
    "Recipe IDs": ARTIFACTS_DIR / "recipe_ids.csv",
    "Embedding metadata": ARTIFACTS_DIR / "embedding_metadata.json",
    "PCA projector": ARTIFACTS_DIR / "pca_projector.pkl",
    "PCA projections (2D)": ARTIFACTS_DIR / "pca_projection.npy",
    "Autoencoder projections (2D)": ARTIFACTS_DIR / "projections_2d.npy",
    "Regression model": ARTIFACTS_DIR / "regressor.joblib",
}

rows = []
for label, path in artifacts.items():
    exists = path.exists()
    size_mb = f"{path.stat().st_size / 1e6:.2f} MB" if exists else "—"
    rows.append({"Artifact": label, "Status": "✅ Present" if exists else "❌ Missing", "Size": size_mb})

import pandas as pd
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Embedding info ────────────────────────────────────────────────────────────
meta_path = ARTIFACTS_DIR / "embedding_metadata.json"
if meta_path.exists():
    st.subheader("Embedding Model")
    meta = json.loads(meta_path.read_text())
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", meta.get("model_name", "—"))
    col2.metric("Dimension", meta.get("embedding_dim", "—"))
    col3.metric("Recipes encoded", f"{meta.get('num_recipes', 0):,}")

# ── PCA diagnostics ───────────────────────────────────────────────────────────
pca_path = ARTIFACTS_DIR / "pca_projector.pkl"
if pca_path.exists():
    st.subheader("PCA Projection")
    import joblib
    pca_model = joblib.load(pca_path)
    ev = pca_model.explained_variance_ratio_
    col1, col2, col3 = st.columns(3)
    col1.metric("PC1 Explained Variance", f"{ev[0]:.2%}")
    col2.metric("PC2 Explained Variance", f"{ev[1]:.2%}")
    col3.metric("Total (PC1+PC2)", f"{sum(ev):.2%}")
    st.caption(
        "A total explained variance of ~15% is typical for sentence-transformer embeddings "
        "projected from 384 → 2 dimensions. The autoencoder may learn a more expressive manifold."
    )

# ── Projection shape check ────────────────────────────────────────────────────
proj_pca = ARTIFACTS_DIR / "pca_projection.npy"
proj_ae = ARTIFACTS_DIR / "projections_2d.npy"

if proj_pca.exists() or proj_ae.exists():
    st.subheader("Projection Shapes")
    shape_rows = []
    if proj_pca.exists():
        arr = np.load(proj_pca)
        shape_rows.append({"Method": "PCA", "Shape": str(arr.shape), "Dtype": str(arr.dtype)})
    if proj_ae.exists():
        arr = np.load(proj_ae)
        shape_rows.append({"Method": "Autoencoder", "Shape": str(arr.shape), "Dtype": str(arr.dtype)})
    st.dataframe(pd.DataFrame(shape_rows), use_container_width=True, hide_index=True)

# ── Run scripts hint ──────────────────────────────────────────────────────────
st.divider()
st.markdown(
    """
    **To generate missing artifacts, run these scripts from the repo root:**
    ```bash
    # Build embeddings (add --limit N for a fast test run)
    PYTHONPATH=src ./venv/bin/python scripts/build_embeddings.py --limit 1000

    # Fit PCA and save projections
    PYTHONPATH=src ./venv/bin/python scripts/fit_pca.py

    # Train autoencoder (uses checkpoint hash — safe to re-run)
    PYTHONPATH=src ./venv/bin/python scripts/train_autoencoder.py --epochs 20
    ```
    """
)
