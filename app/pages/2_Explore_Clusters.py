"""Cluster exploration page.

Loads recipe embeddings + the processed recipe table, fits (or loads) a
k-means model, and presents the user with auto-named cluster cards plus
sample recipes per cluster. The auto-naming uses the TF-IDF + optional
concept-vocabulary path from ``clustering.labels``.

Three "missing artifact" failure modes are handled with clear messages so
teammates running this page locally see what to run, not a stack trace.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components.theme import apply_restaurant_menu_theme
from recipe_discovery.clustering.kmeans import KMeans
from recipe_discovery.clustering.labels import name_clusters
from recipe_discovery.clustering.service import attach_cluster_assignments
from recipe_discovery.data.load import load_processed_recipes
from recipe_discovery.embeddings.store import load_embeddings, load_recipe_ids
from recipe_discovery.evaluation.clustering_eval import cluster_sizes
from recipe_discovery.settings import ARTIFACTS_DIR

apply_restaurant_menu_theme()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Explore Clusters")
st.caption(
    "Browse recipes grouped by semantic similarity. Cluster names are "
    "auto-generated from each group's distinctive vocabulary."
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embeddings...")
def _load_embeddings_cached() -> np.ndarray:
    return load_embeddings()


@st.cache_resource(show_spinner="Loading recipe metadata...")
def _load_recipes_cached() -> pd.DataFrame:
    """Load processed recipe table, aligned by recipe_id with embeddings."""
    df = load_processed_recipes()
    ids = load_recipe_ids()
    # Reindex df to follow the embedding row order so positional joins are safe.
    df = df.set_index("recipe_id").loc[ids].reset_index()
    return df


@st.cache_resource(show_spinner="Fitting k-means...")
def _fit_or_load_kmeans(embeddings_hash: int, n_clusters: int) -> KMeans:
    """Return a fitted KMeans. Prefers the artifact saved by `train_kmeans.py`
    when its ``n_clusters`` matches the user's choice; otherwise fits fresh on
    the embeddings the page already has in memory.

    The ``embeddings_hash`` argument is purely a cache-busting key so changing
    embeddings invalidates the cached model.
    """
    artifact = ARTIFACTS_DIR / "kmeans.joblib"
    if artifact.exists():
        try:
            cached = KMeans.load(artifact)
            if cached.n_clusters == n_clusters:
                return cached
        except (ValueError, FileNotFoundError):
            pass  # fall through to fresh fit
    embeddings = _load_embeddings_cached()
    return KMeans(n_clusters=n_clusters, random_state=42, n_init=5).fit(embeddings)


@st.cache_data(show_spinner="Naming clusters...")
def _name_clusters_cached(
    df_with_clusters: pd.DataFrame,
    text_columns: tuple[str, ...],
    vocabulary: tuple[str, ...] | None,
) -> dict:
    return name_clusters(
        df_with_clusters,
        text_columns=text_columns,
        vocabulary=list(vocabulary) if vocabulary else None,
        top_n=8,
        min_doc_freq=5,
    )


# ---------------------------------------------------------------------------
# Load artifacts (with helpful error messages for missing files)
# ---------------------------------------------------------------------------


def _missing_artifact(message: str, hint: str) -> None:
    st.error(message)
    st.code(hint, language="bash")
    st.stop()


try:
    embeddings = _load_embeddings_cached()
except FileNotFoundError:
    _missing_artifact(
        "Embeddings not found. Run the embeddings build step first.",
        "python scripts/build_embeddings.py",
    )

try:
    recipes_df = _load_recipes_cached()
except FileNotFoundError as exc:
    _missing_artifact(
        f"Processed recipe data missing: {exc}",
        "python scripts/run_data_pipeline.py",
    )
except KeyError:
    # set_index/reindex raised because recipe_ids and the processed CSV don't agree
    _missing_artifact(
        "Recipe IDs in `recipe_ids.csv` don't match the processed recipe table. "
        "Re-run the pipeline so the artifacts are aligned.",
        "python scripts/run_data_pipeline.py && python scripts/build_embeddings.py",
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.subheader("Clustering")
default_k = 8
saved_artifact = ARTIFACTS_DIR / "kmeans.joblib"
if saved_artifact.exists():
    try:
        default_k = KMeans.load(saved_artifact).n_clusters
    except (ValueError, FileNotFoundError):
        pass

n_clusters = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=20,
    value=default_k,
    help="Changing K refits k-means on the fly. The saved artifact "
    "(from `scripts/train_kmeans.py`) is reused when its K matches.",
)

# Choose which text columns to mine for distinctive words. Defaults match
# `data.schema.TEXT_COLUMNS` plus `tags` if present.
candidate_columns = ["name", "description", "ingredients", "steps", "tags"]
available_columns = [c for c in candidate_columns if c in recipes_df.columns]
default_text_columns = [c for c in ("name", "description", "tags") if c in available_columns]

selected_text_columns = st.sidebar.multiselect(
    "Text fields used for cluster naming",
    options=available_columns,
    default=default_text_columns,
)

# A small concept vocabulary for cleaner labels. Mirrors the image-search
# vocabulary so cluster names and detected image categories share language.
DEFAULT_VOCABULARY: tuple[str, ...] = (
    "dessert", "breakfast", "dinner", "appetizer", "snack",
    "soup", "salad", "pizza", "pasta", "burger", "sandwich",
    "taco", "sushi", "curry", "cake", "cookies", "bread", "pie",
    "chicken", "beef", "pork", "seafood", "vegetables", "fruit",
    "italian", "chinese", "japanese", "mexican", "indian", "thai",
    "french", "korean", "grilled", "fried", "roasted",
)
use_vocabulary = st.sidebar.checkbox(
    "Use concept vocabulary for cleaner names",
    value=True,
    help="When on, cluster names favor concept words like 'pasta' or 'desserts'. "
    "When off, names use the raw distinctive tokens (e.g. 'spaghetti, parmesan').",
)


# ---------------------------------------------------------------------------
# Fit + name
# ---------------------------------------------------------------------------

if not selected_text_columns:
    st.warning("Pick at least one text field in the sidebar to name clusters.")
    st.stop()

# Hash key to invalidate caches when embeddings change.
embeddings_hash = int(embeddings.shape[0]) ^ int(embeddings.shape[1])

model = _fit_or_load_kmeans(embeddings_hash, n_clusters)
labels = model.predict(embeddings) if model.labels_ is None else model.labels_

df_with_clusters = attach_cluster_assignments(recipes_df, labels.tolist())
sizes = cluster_sizes(np.asarray(labels))

named = _name_clusters_cached(
    df_with_clusters,
    tuple(selected_text_columns),
    DEFAULT_VOCABULARY if use_vocabulary else None,
)


# ---------------------------------------------------------------------------
# Header summary
# ---------------------------------------------------------------------------

col_a, col_b, col_c = st.columns(3)
col_a.metric("Recipes clustered", f"{len(df_with_clusters):,}")
col_b.metric("Clusters", n_clusters)
if model.inertia_ is not None:
    col_c.metric("Inertia (training)", f"{model.inertia_:,.0f}")


# ---------------------------------------------------------------------------
# Cluster cards
# ---------------------------------------------------------------------------

st.divider()
sorted_clusters = sorted(sizes.keys(), key=lambda c: -sizes[c])

for cid in sorted_clusters:
    info = named.get(cid, {"name": f"Cluster {cid}", "distinctive": [], "matched_concepts": []})
    cluster_name = str(info["name"])
    size = sizes[cid]

    with st.expander(
        f"**{cluster_name}** &nbsp;&nbsp; · &nbsp;&nbsp; {size:,} recipes "
        f"({size / len(df_with_clusters) * 100:.1f}%)"
    ):
        # Distinctive vocabulary chips
        words = info.get("distinctive", [])
        if words:
            st.caption("Top distinctive words")
            chips = " &nbsp; ".join(
                f"`{w}` *(score {s:.1f})*" for w, s in words[:6]
            )
            st.markdown(chips)

        matched = info.get("matched_concepts", [])
        if matched:
            st.caption("Matched concepts")
            st.markdown(", ".join(f"**{c}**" for c in matched))

        # Sample recipes
        cluster_rows = df_with_clusters[df_with_clusters["cluster"] == cid]
        if "name" in cluster_rows.columns and len(cluster_rows) > 0:
            sample = cluster_rows.sample(
                n=min(5, len(cluster_rows)),
                random_state=cid,  # stable per-cluster sample for screenshots
            )
            st.caption("Sample recipes")
            for _, row in sample.iterrows():
                title = str(row.get("name", "Untitled"))
                description = str(row.get("description", ""))[:140]
                st.markdown(f"- **{title}**" + (f" — {description}…" if description else ""))


# ---------------------------------------------------------------------------
# Footer note
# ---------------------------------------------------------------------------

st.divider()
st.caption(
    "Names are generated automatically using log-odds TF-IDF over the selected "
    "text fields, optionally projected onto a hand-curated concept vocabulary. "
    "No manual labelling required."
)
