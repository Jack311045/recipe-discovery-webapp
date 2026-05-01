"""Model Diagnostics — shows PCA explained variance, embedding info, and artifact status."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import json

import numpy as np
import pandas as pd
import streamlit as st

from app.components.theme import apply_restaurant_menu_theme
from recipe_discovery.settings import ARTIFACTS_DIR

apply_restaurant_menu_theme()
st.markdown(
    """
    <style>
    .diagnostic-model-card {
        min-height: 6.25rem;
        padding: 0.62rem 0.7rem;
        border: 1px solid var(--menu-border);
        border-radius: 0.75rem;
        background: rgba(255, 250, 241, 0.94);
        box-shadow: 0 2px 8px rgba(88, 60, 40, 0.07);
    }

    .diagnostic-model-label {
        margin-bottom: 0.34rem;
        color: #5b4434;
        font-size: 0.86rem;
        font-weight: 650;
        line-height: 1.2;
    }

    .diagnostic-model-name {
        color: #3b2f2a;
        font-size: clamp(1.05rem, 1.7vw, 1.6rem);
        font-weight: 650;
        line-height: 1.18;
        overflow-wrap: anywhere;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🔬 Model Diagnostics")
st.caption("Verify that all artifacts are present and inspect model characteristics.")

# ── Artifact status ──────────────────────────────────────────────────────────
st.subheader("Artifact Status")

artifacts = {
    "Embeddings": ARTIFACTS_DIR / "recipe_embeddings.npy",
    "Recipe IDs": ARTIFACTS_DIR / "recipe_ids.csv",
    "Embedding metadata": ARTIFACTS_DIR / "embedding_metadata.json",
    "SigLIP embeddings": ARTIFACTS_DIR / "recipe_embeddings_siglip.npy",
    "SigLIP recipe IDs": ARTIFACTS_DIR / "recipe_ids_siglip.csv",
    "SigLIP metadata": ARTIFACTS_DIR / "embedding_metadata_siglip.json",
    "PCA projector": ARTIFACTS_DIR / "pca_projector.pkl",
    "PCA projections (2D)": ARTIFACTS_DIR / "pca_projection.npy",
    "Regression model": ARTIFACTS_DIR / "regressor.joblib",
}

rows = []
for label, path in artifacts.items():
    exists = path.exists()
    size_mb = f"{path.stat().st_size / 1e6:.2f} MB" if exists else "—"
    rows.append(
        {
            "Artifact": label,
            "Status": "✅ Present" if exists else "❌ Missing",
            "Size": size_mb,
        }
    )

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Embedding info ────────────────────────────────────────────────────────────
meta_path = ARTIFACTS_DIR / "embedding_metadata.json"
if meta_path.exists():
    st.subheader("Embedding Model")
    meta = json.loads(meta_path.read_text())
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="diagnostic-model-card">
                <div class="diagnostic-model-label">Model</div>
                <div class="diagnostic-model-name">{meta.get("model_name", "—")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
        "projected from 384 → 2 dimensions."
    )

# ── Projection shape check ────────────────────────────────────────────────────
proj_pca = ARTIFACTS_DIR / "pca_projection.npy"

if proj_pca.exists():
    st.subheader("Projection Shapes")
    shape_rows = []
    arr = np.load(proj_pca)
    shape_rows.append({"Method": "PCA", "Shape": str(arr.shape), "Dtype": str(arr.dtype)})
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

    ```
    """
)
