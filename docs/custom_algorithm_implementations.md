# Custom / From-Scratch Algorithm Implementations

This repo does include project-owned algorithm implementations that are more
than thin wrappers around a library estimator. The clearest example is the
PyTorch k-means clustering implementation.

## Summary

| Component | Custom or wrapper? | Location | Notes |
|---|---|---|---|
| K-means clustering | Custom/from scratch | `src/recipe_discovery/clustering/kmeans.py` | Own PyTorch implementation of initialization, assignment, centroid update, convergence, restarts, and persistence |
| Autoencoder reducer | Custom PyTorch model/training loop | `src/recipe_discovery/reduction/autoencoder.py`, `scripts/train_autoencoder.py` | Uses PyTorch layers/optimizers, but architecture and training loop are implemented in the repo |
| Cosine retrieve-all scoring | Custom NumPy utility | `src/recipe_discovery/retrieval/similarity.py` | Computes cosine scores directly against the matrix rather than using a library search API |
| PCA reducer | Library-backed wrapper | `src/recipe_discovery/reduction/pca.py` | Wraps `sklearn.decomposition.PCA` |
| Regression model | Library-backed wrapper | `src/recipe_discovery/models/regression.py` | Wraps sklearn `StandardScaler` plus `Ridge` pipeline |
| Classifier | Library-backed wrapper | `src/recipe_discovery/models/classification.py` | Wraps sklearn `LogisticRegression` |
| Optional NN index | Library-backed wrapper | `src/recipe_discovery/embeddings/index.py` | Wraps sklearn `NearestNeighbors` |

## 1. From-Scratch PyTorch K-Means

Primary file:

```text
src/recipe_discovery/clustering/kmeans.py
```

Training script:

```text
scripts/train_kmeans.py
```

Tests:

```text
tests/test_kmeans.py
```

This is the main answer to the question. The project does not call
`sklearn.cluster.KMeans`. Instead, it defines its own `KMeans` class and
implements the core clustering mechanics with `torch` and `numpy`.

### What Is Implemented Directly

The custom `KMeans` class implements:

1. Input validation for `n_clusters`, `max_iter`, `n_init`, and `init_method`.
2. Device selection with CPU/CUDA support.
3. Random centroid initialization.
4. K-means++ centroid initialization.
5. Lloyd assignment step.
6. Lloyd centroid update step.
7. Empty-cluster reseeding.
8. Multiple restarts with `n_init`.
9. Best-run selection by inertia.
10. Prediction for new points.
11. Distance transform against learned centroids.
12. Inertia scoring on arbitrary data.
13. Save/load of the fitted centroid payload.
14. Elbow-curve helper over multiple K values.

### Algorithm Flow

At a high level, `KMeans.fit()` does:

```text
input embeddings
  -> convert to torch.float32 tensor
  -> repeat n_init times:
       initialize centroids
       repeat until max_iter or centroid shift < tol:
         assign every point to nearest centroid
         recompute each centroid as the mean of assigned points
         reseed any empty cluster
       compute final inertia
  -> keep the run with lowest inertia
  -> store centroids_, labels_, inertia_, n_iter_
```

### K-Means++ Initialization

The method:

```text
KMeans._init_kmeans_pp()
```

implements k-means++ manually:

1. Pick the first centroid uniformly at random.
2. Compute each point's squared distance to its nearest selected centroid.
3. Sample the next centroid with probability proportional to that squared
   distance.
4. Update each point's closest-centroid distance.
5. Repeat until `n_clusters` centroids are selected.

This is implemented with:

```text
torch.randint
torch.cdist
torch.multinomial
torch.minimum
```

The important point: PyTorch is used as the tensor engine, but the actual
k-means++ procedure is implemented in this repo.

### Assignment Step

The method:

```text
KMeans._assign()
```

is the E-step-style part of Lloyd's algorithm:

```text
distance_matrix = squared Euclidean distances from every point to every centroid
label_i = nearest centroid index for point i
min_sq_i = squared distance to that nearest centroid
```

The implementation uses `torch.cdist(x, centroids) ** 2` and then takes the
minimum distance along the centroid axis.

### Centroid Update Step

The method:

```text
KMeans._update()
```

is the M-step-style part of Lloyd's algorithm:

```text
new_centroid_k = mean(points assigned to cluster k)
```

It uses `torch.index_add_` to accumulate point sums and cluster counts. That is
the repo-owned centroid update logic; no sklearn estimator is doing it.

### Empty-Cluster Handling

The implementation explicitly handles empty clusters. If a cluster receives no
points during an update, the code reseeds that centroid to a faraway data point:

```text
find empty cluster indices
compute distance from each point to the non-empty centroids
choose farthest points as replacement centroids
```

This matters because empty clusters are a common k-means failure mode,
especially when `n_clusters` is high or data has duplicate/near-duplicate
points.

### Multiple Restarts and Inertia

`KMeans.fit()` runs `_run_once()` `n_init` times, using deterministic seeds:

```text
random_state + run_index
```

Each run returns:

```text
centroids
labels
inertia
n_iter
```

The fitted model keeps the run with the lowest inertia:

```text
inertia = sum of squared distances to assigned centroids
```

This mirrors the behavior people expect from mature k-means libraries, but the
loop is implemented directly here.

### Persistence

The fitted model saves a plain payload with:

```text
n_clusters
max_iter
tol
n_init
init_method
random_state
centroids_
inertia_
n_iter_
```

It uses `joblib` only for serialization. The clustering math is not delegated to
joblib or sklearn.

### Integration With the App

The training entry point is:

```bash
python scripts/train_kmeans.py
```

It reads:

```text
configs/clustering.yaml
data/artifacts/recipe_embeddings.npy
```

It writes:

```text
data/artifacts/kmeans.joblib
data/artifacts/kmeans_metadata.json
```

The metadata explicitly labels the model type as:

```text
from_scratch_pytorch_kmeans
```

The app can then load cluster labels for exploration and map views.

### Validation

`tests/test_kmeans.py` verifies behavior that would be easy to get wrong in a
manual implementation:

1. Fit/predict API shape.
2. Centroid shape and label range.
3. `predict(x)` matches labels from `fit(x)`.
4. Distance transform shape and nonnegative values.
5. Inertia and iteration count are set.
6. Recovery of well-separated Gaussian blobs with ARI greater than 0.9.
7. Inertia does not increase across Lloyd iterations.
8. More restarts improve or match final inertia.
9. K-means++ beats or matches random initialization on average.
10. Empty clusters are reseeded.
11. `score_inertia()` works on separate data.
12. Elbow inertia decreases as K increases.
13. Same seed gives reproducible centroids.
14. NumPy and torch tensor inputs both work.
15. Save/load roundtrip preserves predictions.

The tests use sklearn only for validation metrics such as adjusted Rand index,
not for fitting the clusters.

## 2. Custom PyTorch Autoencoder Reducer

Primary file:

```text
src/recipe_discovery/reduction/autoencoder.py
```

Training script:

```text
scripts/train_autoencoder.py
```

This is also a custom implementation, although it naturally uses PyTorch neural
network primitives. It is not a wrapper around `sklearn` or a prebuilt
autoencoder estimator.

### Architecture

The repo defines an `Autoencoder(nn.Module)` with:

```text
encoder:
  input_dim -> 512 -> 256 -> 128 -> latent_dim

decoder:
  latent_dim -> 128 -> 256 -> 512 -> input_dim
```

Default config:

```text
input_dim = 768
latent_dim = 2
hidden_dims = (512, 256, 128)
dropout_rate = 0.1
```

The training script overrides `input_dim` from the actual embedding matrix:

```text
input_dim = embeddings.shape[1]
```

so it can train on whichever embedding store is supplied.

### Training Loop

`scripts/train_autoencoder.py` owns the training loop:

1. Load `.npy` embeddings.
2. L2-normalize embeddings.
3. Randomly split 80/20 train/validation.
4. Build PyTorch `DataLoader` objects.
5. Create the custom `AutoencoderReducer`.
6. Train with `AdamW`.
7. Use `MSELoss` reconstruction objective.
8. Optionally add Gaussian noise for denoising training.
9. Track validation loss.
10. Save the best checkpoint.
11. Stop early when validation loss stops improving.
12. Generate and save 2D latent projections.

The model uses PyTorch for tensor execution, layers, and optimization, but the
project owns the architecture, data split, denoising option, checkpointing, and
early-stopping logic.

### Output Artifacts

The training script writes:

```text
data/artifacts/autoencoder_<hash>.pt
data/artifacts/projections_2d.npy
```

The hash is based on the embedding path and config so cached checkpoints can be
reused when the same training setup is repeated.

### Caveat

Compared with k-means, the autoencoder has less direct test coverage in the
current test suite. It is still a custom implementation, but k-means is the
stronger example to cite because it has extensive algorithm-specific tests.

## 3. Custom Cosine Retrieve-All Scoring

Primary files:

```text
src/recipe_discovery/retrieval/similarity.py
src/recipe_discovery/retrieval/service.py
```

This is smaller than k-means, but it is another place where the project does not
simply call a library search API at runtime.

The function:

```text
cosine_similarity(query, matrix)
```

computes:

```text
(matrix @ query) / (||matrix rows|| * ||query||)
```

with NumPy. `RetrievalService._search_candidates_for_vector()` calls this
against the entire loaded embedding matrix:

```text
scores = cosine_similarity(query=query_vec, matrix=embeddings)
```

Then it uses:

```text
np.argpartition
np.argsort
```

to select and rank a candidate pool before applying filters.

This is not a full ML algorithm like k-means, but it is a project-owned
implementation of the live retrieve-all scoring path. The optional
`sklearn.neighbors.NearestNeighbors` index exists only as an offline/optional
artifact, not as the default online search path.

## What Should Not Be Claimed as From Scratch

These parts are useful and important, but they are intentionally library-backed:

### PCA

Location:

```text
src/recipe_discovery/reduction/pca.py
```

Uses:

```text
sklearn.decomposition.PCA
```

This is a wrapper with project-specific normalization and persistence.

### Regression

Location:

```text
src/recipe_discovery/models/regression.py
```

Uses:

```text
sklearn.pipeline.Pipeline
sklearn.preprocessing.StandardScaler
sklearn.linear_model.Ridge
```

This is a baseline model wrapper, not a from-scratch optimizer.

### Classification

Location:

```text
src/recipe_discovery/models/classification.py
```

Uses:

```text
sklearn.linear_model.LogisticRegression
```

### Optional Nearest-Neighbor Index

Location:

```text
src/recipe_discovery/embeddings/index.py
```

Uses:

```text
sklearn.neighbors.NearestNeighbors
```

The live retrieval path uses custom direct cosine scoring, but this saved index
artifact itself is library-backed.

## Best Short Answer for Reports / Presentation

The project includes a from-scratch PyTorch implementation of k-means clustering
in `src/recipe_discovery/clustering/kmeans.py`. It does not use
`sklearn.cluster.KMeans`; it implements k-means++ initialization, Lloyd
assignment and centroid-update steps, empty-cluster reseeding, multiple random
restarts, inertia scoring, prediction, transform, and save/load logic. The
training script is `scripts/train_kmeans.py`, and the behavior is validated in
`tests/test_kmeans.py`.

The project also defines a custom PyTorch autoencoder for nonlinear 2D embedding
projection in `src/recipe_discovery/reduction/autoencoder.py` with a hand-written
training loop in `scripts/train_autoencoder.py`. This uses PyTorch primitives,
but the architecture and training procedure are project-owned rather than a
prebuilt sklearn-style estimator.

