# Vibe Coding Outline: Dimensionality Reduction Module
### `src/recipe_discovery/reduction/`

> **Strategy:** Build PCA first as a fast baseline to validate the full pipeline end-to-end.
> Then layer in the Autoencoder for richer, non-linear projections. Never block clustering
> or integration work on the Autoencoder being trained.

---

## Phase 0 — Module Scaffolding

**Goal:** Establish the file structure and shared interfaces before writing any math.

- [ ] Create `reduction/__init__.py` — export `PCAReducer`, `AutoencoderReducer`, and a unified `reduce()` dispatcher
- [ ] Define a shared `BaseReducer` abstract class with two methods:
  - `fit(embeddings: np.ndarray) -> None`
  - `transform(embeddings: np.ndarray) -> np.ndarray`  *(shape: `[N, 2]`)*
- [ ] Decide and document the **expected input shape**: `[N, 768]` (sentence-transformer default) — confirm this against your `RecipeEncoder` output before writing anything else
- [ ] Add a `reduction/config.py` for hyperparameters (n_components, hidden dims, learning rate, etc.) so nothing is hardcoded

### 0.1 L2 Normalization (Must-Have Pre-Processing Step)

Add a `normalize_embeddings()` utility to `utils.py` and call it as the **first step** inside both `PCAReducer.fit()` and `AutoencoderReducer.fit()`, before any other processing:

```python
# reduction/utils.py
def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to unit sphere before reduction."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, a_min=1e-10, a_max=None)
```

**Why this is non-negotiable:** Sentence-transformer embeddings are cosine-similarity vectors that live on a hypersphere. PCA and the Autoencoder both assume Euclidean geometry. Without normalization, high-magnitude embeddings (long recipes with many ingredients) will dominate the projection axes and your 2D space will reflect recipe *length* more than recipe *meaning*. Normalize once here; do not rely on consuming modules to do it.

> 💡 **Vibe check:** Run a quick sanity script that loads 500 embeddings, prints their shape, and checks `np.linalg.norm(embeddings, axis=1)` — all values should be ~1.0 after normalization. Do this *before* Phase 1. It takes 5 minutes and prevents hours of debugging.

---

## Phase 1 — PCA Baseline (Do This First)

**Goal:** A working 2D projection in < 1 hour. Validates the full pipeline cheaply.

### 1.1 Implement `PCAReducer`
- Wrap `sklearn.decomposition.PCA(n_components=2)`
- Call `normalize_embeddings()` at the top of `fit()` before passing to PCA
- Implement `fit()` and `transform()` per `BaseReducer`
- Add `fit_transform()` convenience method
- Persist the fitted PCA object with `joblib.dump()` so it can be reloaded without retraining

### 1.2 Explained Variance Check
- After fitting, log `pca.explained_variance_ratio_` — you want the two components to explain **> 15% combined** on recipe embeddings; below that, your embeddings may need inspection
- Plot a scree plot (even just locally with matplotlib) to see how many components you'd need for 80–90% variance — useful context for the Autoencoder bottleneck size later
- **Export the full-dataset projections here** — the clustering module will use these for silhouette scoring to tune K (see Phase 4 handoff contract)

### 1.3 Smoke Test
- Load 1,000 recipe embeddings → `normalize` → `fit_transform` → assert output shape is `[1000, 2]`
- Scatter plot the 2D output colored by a simple heuristic (e.g., recipe category or rating bucket) — if clusters are visually coherent, PCA is working

> ✅ **Gate:** Only move to Phase 2 once PCA projections are loaded by `KMeans` and a scatter plot renders in Streamlit (even a static one). This confirms the full pipeline is wired.

---

## Phase 2 — Autoencoder Architecture & Training

**Goal:** A non-linear 2D projection that preserves more semantic structure than PCA.

### 2.1 Architecture Design

```
Input [768]  ← L2-normalized before entering
  → Linear(768, 512) + ReLU + BatchNorm
  → Dropout(0.1)                           ← after first encoder layer
  → Linear(512, 256) + ReLU + BatchNorm
  → Linear(256, 128) + ReLU
  → Linear(128, 2)                         ← BOTTLENECK (no activation, unconstrained)
  → Linear(2, 128)   + ReLU
  → Linear(128, 256) + ReLU
  → Linear(256, 512) + ReLU
  → Linear(512, 768)                       ← Reconstructed output
```

**Architecture notes:**
- **Dropout(0.1)** sits after the first encoder layer only — early-layer dropout regularizes without disrupting the fine-grained structure the deeper layers need to learn. Do not add it near the bottleneck.
- **No activation on the bottleneck** — unconstrained 2D space lets clusters spread naturally. Adding ReLU here is the single most common mistake in visualization autoencoders.
- **BatchNorm in encoder only** — recipe embeddings have high variance across ingredient domains; BN stabilizes encoder training. Skip it in the decoder to avoid reconstruction artifacts.
- **Loss:** `nn.MSELoss()` on `(reconstructed, original)` — no need for anything exotic.

### 2.2 Training Setup

- DataLoader: Wrap your embeddings in a `TensorDataset`; use `batch_size=256`, `shuffle=True`
- Optimizer: **`AdamW(lr=1e-3, weight_decay=1e-4)`** — use AdamW instead of Adam; weight decay prevents encoder weights from growing large and distorting the latent space, with negligible training cost
- Scheduler: `ReduceLROnPlateau(patience=5, factor=0.5)` on validation loss
- Epochs: Start with **50 epochs**; watch the loss curve — if it plateaus before 30, your architecture may be too small
- Save best checkpoint by validation loss (80/20 train/val split on embeddings)

### 2.3 Denoising Mode (Should-Have Enhancement)

Add a `--denoise` flag to the training script that injects Gaussian noise into inputs during training only:

```python
# Inside training loop, if denoising mode is on:
noisy_input = embeddings + torch.randn_like(embeddings) * noise_std  # noise_std=0.05
reconstructed = model(noisy_input)
loss = criterion(reconstructed, embeddings)  # target is still the CLEAN embedding
```

**Why it's worth it:** Denoising forces the encoder to learn smoother, more robust latent representations rather than memorizing exact embedding coordinates. The practical effect is better cluster separability in the 2D space — desserts and savory dishes will pull apart more cleanly. Use `noise_std=0.05` as a starting point; increase to `0.1` if clusters still look muddy after normal training.

### 2.4 Training Script (`scripts/train_autoencoder.py`)

```python
# Suggested CLI args
--embeddings_path   # path to precomputed .npy embeddings
--output_dir        # where to save model weights + 2D projections
--epochs            # default 50
--batch_size        # default 256
--lr                # default 1e-3
--weight_decay      # default 1e-4
--denoise           # flag to enable denoising mode
--noise_std         # default 0.05, only used if --denoise is set
--device            # cuda / mps / cpu (auto-detect if not passed)
```

- Save two artifacts:
  1. `autoencoder_weights.pt` — full model state dict
  2. `projections_2d.npy` — precomputed 2D coords for the entire dataset (avoids inference latency at serve time)

### 2.5 Implement `AutoencoderReducer`

- `fit()` → runs the training loop (or loads weights if checkpoint exists — hash embeddings path + config to decide)
- `transform()` → encoder forward pass only (no decoder needed at inference); apply `normalize_embeddings()` first
- Add a `load_from_checkpoint(path)` classmethod for the Streamlit app to use

---

## Phase 3 — Hyperparameter Tuning & Validation

**Goal:** Know which knobs to turn, in what order, and how to confirm the 2D space is actually good.

### 3.1 Tuning Table

| Parameter | Start Here | Tune If... |
|---|---|---|
| Bottleneck dim | 2 (fixed for viz) | Clusters look collapsed → try 3D then project |
| Hidden dims | `[512, 256, 128]` | Loss stalls early → increase; slow training → decrease |
| Learning rate | `1e-3` | Loss oscillates → lower to `3e-4`; plateaus fast → raise to `3e-3` |
| Weight decay | `1e-4` | Overfitting (train << val loss) → increase to `1e-3` |
| Dropout | `0.1` (encoder layer 1) | Overfitting persists → add second Dropout(0.1) after layer 2 |
| Noise std (denoising) | `0.05` | Clusters still muddy → increase to `0.1`; reconstruction suffers → lower to `0.02` |
| Batch size | `256` | GPU OOM → halve it; slow on CPU → increase to 512 |
| Epochs | `50` | Val loss still dropping → extend; overfitting → reduce + increase dropout |
| PCA n_components | `2` | Only change for intermediate steps, never the final output |

### 3.2 PCA–Autoencoder Consistency Check (Nice-to-Have Validation)

After training the Autoencoder, run a consistency check to confirm it has learned something meaningfully *different* from PCA (not just a worse version of it):

```python
# reduction/utils.py
def procrustes_similarity(pca_2d: np.ndarray, ae_2d: np.ndarray) -> float:
    """Align AE projections to PCA via Procrustes and return residual similarity score.
    Score near 1.0 = AE learned same structure as PCA (likely undertrained).
    Score near 0.0 = totally different (likely overtrained or broken).
    Healthy range: 0.3–0.7."""
    from scipy.spatial import procrustes
    _, _, disparity = procrustes(pca_2d, ae_2d)
    return 1 - disparity
```

Also check that KMeans cluster assignments on PCA vs AE projections have **40–70% overlap** (via adjusted Rand index). Below 40% suggests the AE is not preserving semantic structure; above 70% suggests it's just replicating PCA and the non-linear training wasn't useful.

### 3.3 Signs Your 2D Space is Healthy
- KMeans clusters match culinary intuition (desserts group together, quick weeknight meals cluster separately)
- Points do not collapse to a tight ball — if they do, learning rate is too high or BatchNorm is misconfigured
- PCA and AE projections look visually different but structurally similar in cluster layout
- Procrustes similarity is in the 0.3–0.7 range
- Cluster assignment overlap (ARI) between PCA and AE is 40–70%

---

## Phase 4 — Integration with `RetrievalService`

**Goal:** The retrieval pipeline can attach 2D coords to every returned recipe, and consuming modules receive a clearly documented contract.

### 4.1 Precompute & Store Projections
- After training, run `transform()` on the full dataset and save as `projections_2d.npy`
- Store alongside your recipe DataFrame so that each recipe ID maps to a `(x, y)` coordinate
- Recommended: add `x_proj` and `y_proj` columns directly to your processed recipes Parquet/CSV

### 4.2 `RetrievalService` Changes
- On init, load the projections array (or the columns from the DataFrame)
- After a search returns top-K recipe indices, attach their `(x, y)` coords to the result objects
- Add a method `get_all_projections() -> pd.DataFrame` for the Streamlit scatter plot (it needs *all* points as background, not just top-K)

### 4.3 Reducer Loader Utility with Caching
```python
# reduction/utils.py
def load_reducer(method: str = "autoencoder", embeddings_hash: str = None) -> BaseReducer:
    """
    Load fitted PCA or Autoencoder from disk.
    Falls back to PCA if AE weights not found.
    Uses embeddings_hash + config hash to skip retraining if a matching checkpoint exists.
    """
```
Hash = `sha256(embeddings_path + config_str)[:8]`. Save checkpoints as `autoencoder_{hash}.pt`. On load, if the hash matches an existing checkpoint, skip training entirely. This prevents accidental full retrains during development when you tweak unrelated config values.

---

## Handoff Contract — What Consuming Modules Should Expect

> Document this clearly so `clustering/` and `retrieval/` don't make assumptions that break silently.

**`reduction/` guarantees:**
- Output shape is always `[N, 2]`, dtype `float32`
- Input embeddings are L2-normalized internally — consuming modules do **not** need to normalize before calling `transform()`
- Projections are saved to `projections_2d.npy` and as `x_proj`/`y_proj` DataFrame columns after any training run

**`clustering/` is responsible for:**
- Re-normalizing the 2D projections before KMeans if cosine distance is desired (the 2D space is Euclidean, so this is optional but worth testing)
- Running KMeans with multiple initializations (`n_init=10` minimum) — not the reducer's job
- Using the exported PCA projections (available after Phase 1) for silhouette-based K tuning — do not wait for the Autoencoder

**`retrieval/` is responsible for:**
- Temperature scaling on similarity scores — unrelated to 2D coords
- MMR diversity re-ranking — unrelated to 2D coords
- Loading `x_proj`/`y_proj` from the DataFrame to attach to search results

---

## Suggested Vibe Coding Session Order

```
Session 1  →  Phase 0 + Phase 1 (scaffolding + L2 norm + PCA working end-to-end)
Session 2  →  Phase 4.1–4.2 (integration, so clustering/UI work can start in parallel)
Session 3  →  Phase 2.1–2.2 (Autoencoder architecture + AdamW + Dropout + training loop)
Session 4  →  Phase 2.3–2.5 (denoising mode + training script + checkpointing)
Session 5  →  Phase 3 (tuning + consistency checks)
Session 6  →  Phase 4.3 + final swap from PCA → Autoencoder in the app
```

---

## Key Files Summary

```
src/recipe_discovery/reduction/
├── __init__.py          # exports PCAReducer, AutoencoderReducer, load_reducer
├── base.py              # BaseReducer ABC
├── pca.py               # PCAReducer (Phase 1)
├── autoencoder.py       # AutoencoderReducer + nn.Module definition (Phase 2)
├── config.py            # all hyperparameters (lr, weight_decay, noise_std, etc.)
└── utils.py             # normalize_embeddings, load_reducer, procrustes_similarity

scripts/
└── train_autoencoder.py # CLI training script with --denoise flag (Phase 2)
```
