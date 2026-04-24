"""K-means clustering on recipe embeddings, implemented from scratch in PyTorch.

The public API stays compatible with the existing stub — ``KMeans(n_clusters=...,
random_state=...)`` still works — but the implementation is upgraded to:

- k-means++ initialization (Arthur & Vassilvitskii, 2007) instead of uniform random.
- Squared Euclidean distance. On L2-normalized sentence-transformer outputs, this
  is equivalent (up to a constant) to minimizing negative cosine similarity.
- Empty-cluster reseeding to the data point currently farthest from any non-empty
  centroid; prevents the algorithm from silently collapsing to < K effective
  clusters when an initialization is unlucky.
- ``n_init`` random restarts; the run with lowest inertia is kept (matches
  scikit-learn semantics).
- ``inertia_`` and ``n_iter_`` attributes for elbow plots and convergence
  diagnostics.
- ``save`` / ``load`` helpers that serialize centroids to joblib, mirroring
  ``RecipeRegressor`` so artifacts are cross-machine portable (CPU-only teammates
  can load models trained with GPU).

References
----------
- Lloyd (1982), "Least squares quantization in PCM"
- Arthur & Vassilvitskii (2007), "k-means++: The Advantages of Careful Seeding"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from recipe_discovery.utils.io import ensure_parent_dir


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a device spec, including the ``"auto"`` sentinel."""
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class KMeans:
    """K-means clustering implemented from scratch in PyTorch.

    Parameters
    ----------
    n_clusters : int
        Number of clusters ``K``.
    max_iter : int, default=100
        Maximum Lloyd iterations per run.
    tol : float, default=1e-4
        Convergence tolerance on the Frobenius norm of the centroid shift.
    n_init : int, default=10
        Number of random restarts; the run with lowest inertia is kept.
    init_method : {"k-means++", "random"}, default="k-means++"
        Centroid initialization strategy.
    random_state : int, default=42
        Base seed; restart ``i`` uses ``random_state + i``.
    device : str | torch.device | None, default=None
        PyTorch device. ``None`` or ``"auto"`` selects CUDA when available,
        else CPU.
    verbose : bool, default=False
        Print per-iteration inertia and centroid shift.

    Attributes
    ----------
    centroids_ : np.ndarray of shape (n_clusters, n_features)
        Set after :meth:`fit`. Stored as NumPy so artifacts are device-agnostic.
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignment for each training point.
    inertia_ : float
        Sum of squared distances from each point to its assigned centroid.
    n_iter_ : int
        Lloyd iterations used by the best (lowest-inertia) run.
    """

    _VALID_INIT_METHODS = ("k-means++", "random")

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        tol: float = 1e-4,
        n_init: int = 10,
        init_method: str = "k-means++",
        random_state: int = 42,
        device: str | torch.device | None = None,
        verbose: bool = False,
    ) -> None:
        if n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {n_clusters}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        if n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {n_init}")
        if init_method not in self._VALID_INIT_METHODS:
            raise ValueError(
                f"init_method must be one of {self._VALID_INIT_METHODS}, got {init_method!r}"
            )

        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.init_method = init_method
        self.random_state = int(random_state)
        self.device = _resolve_device(device)
        self.verbose = bool(verbose)

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    # ------------------------------------------------------------------ init

    def _init_random(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """Pick ``K`` distinct points uniformly at random as initial centroids."""
        n = x.shape[0]
        idx = torch.randperm(n, generator=generator, device=x.device)[: self.n_clusters]
        return x[idx].clone()

    def _init_kmeans_pp(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """k-means++ seeding: first centroid uniform random, each subsequent centroid
        sampled with probability proportional to ``D(x)^2``, where ``D(x)`` is the
        distance to the nearest already-chosen centroid."""
        n, d = x.shape
        centroids = torch.empty(self.n_clusters, d, device=x.device, dtype=x.dtype)

        first_idx = torch.randint(0, n, (1,), generator=generator, device=x.device)
        centroids[0] = x[first_idx]

        # Running minimum squared distance from each point to any chosen centroid.
        closest_sq = torch.cdist(x, centroids[0:1]).squeeze(1) ** 2  # (n,)

        for k in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                # Degenerate case: every point coincides with a chosen centroid.
                next_idx = torch.randint(0, n, (1,), generator=generator, device=x.device)
            else:
                probs = closest_sq / total
                next_idx = torch.multinomial(probs, 1, generator=generator)
            centroids[k] = x[next_idx]
            new_sq = torch.cdist(x, centroids[k : k + 1]).squeeze(1) ** 2
            closest_sq = torch.minimum(closest_sq, new_sq)

        return centroids

    # --------------------------------------------------------------- Lloyd

    @staticmethod
    def _assign(
        x: torch.Tensor, centroids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """E-step: assign each point to its nearest centroid under squared Euclidean.

        Returns
        -------
        labels : (n,) int64 tensor
        min_sq : (n,) float tensor of squared distances to the assigned centroid
        """
        dists_sq = torch.cdist(x, centroids) ** 2  # (n, K)
        min_sq, labels = dists_sq.min(dim=1)
        return labels, min_sq

    def _update(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """M-step: each centroid becomes the mean of its assigned points.

        Empty clusters are reseeded to the data point currently farthest from any
        non-empty centroid.
        """
        k_total, d = self.n_clusters, x.shape[1]
        new_centroids = torch.zeros(k_total, d, device=x.device, dtype=x.dtype)
        counts = torch.zeros(k_total, device=x.device, dtype=x.dtype)

        new_centroids.index_add_(0, labels, x)
        counts.index_add_(0, labels, torch.ones_like(labels, dtype=x.dtype))

        non_empty = counts > 0
        new_centroids[non_empty] /= counts[non_empty].unsqueeze(1)

        empty_idx = torch.where(~non_empty)[0]
        if len(empty_idx) > 0:
            if non_empty.sum().item() == 0:
                # Pathological: nothing to re-seed from. Should be unreachable with K <= n.
                raise RuntimeError(
                    "All clusters empty after update; check that n_samples >= n_clusters."
                )
            dists_to_non_empty = torch.cdist(x, new_centroids[non_empty])
            min_dist, _ = dists_to_non_empty.min(dim=1)
            for k in empty_idx.tolist():
                farthest = int(min_dist.argmax().item())
                new_centroids[k] = x[farthest]
                # Prevent two empty clusters from grabbing the same point.
                min_dist[farthest] = -1.0

        return new_centroids

    def _run_once(
        self, x: torch.Tensor, run_seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        """One Lloyd run; returns (centroids, labels, inertia, n_iter)."""
        generator = torch.Generator(device=x.device).manual_seed(run_seed)

        if self.init_method == "k-means++":
            centroids = self._init_kmeans_pp(x, generator)
        else:
            centroids = self._init_random(x, generator)

        n_iter = 0
        for it in range(1, self.max_iter + 1):
            n_iter = it
            labels, sq_dists = self._assign(x, centroids)
            new_centroids = self._update(x, labels)
            shift = float(torch.linalg.norm(new_centroids - centroids).item())

            if self.verbose:
                print(
                    f"    iter {it:>3d}  inertia={float(sq_dists.sum().item()):.4f}  "
                    f"shift={shift:.6f}"
                )

            centroids = new_centroids
            if shift < self.tol:
                break

        labels, sq_dists = self._assign(x, centroids)
        inertia = float(sq_dists.sum().item())
        return centroids, labels, inertia, n_iter

    # --------------------------------------------------------------- API

    def fit(self, x: np.ndarray | torch.Tensor) -> KMeans:
        """Run k-means ``n_init`` times; keep the lowest-inertia result.

        Parameters
        ----------
        x : np.ndarray or torch.Tensor of shape (n_samples, n_features)
        """
        x_t = self._as_tensor(x)
        if x_t.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={x_t.shape[0]} must be >= n_clusters={self.n_clusters}"
            )

        best_inertia = float("inf")
        best_centroids: torch.Tensor | None = None
        best_labels: torch.Tensor | None = None
        best_iter = 0

        for run in range(self.n_init):
            if self.verbose:
                print(f"Run {run + 1}/{self.n_init}")
            centroids, labels, inertia, n_iter = self._run_once(
                x_t, self.random_state + run
            )
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_iter = n_iter

        assert best_centroids is not None
        assert best_labels is not None

        # Store centroids as NumPy so saved artifacts are device-agnostic.
        self.centroids_ = best_centroids.detach().cpu().numpy()
        self.labels_ = best_labels.detach().cpu().numpy()
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        return self

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Assign cluster labels to new data points."""
        self._check_fitted()
        x_t = self._as_tensor(x)
        centroids_t = torch.from_numpy(self.centroids_).to(
            device=x_t.device, dtype=x_t.dtype
        )
        labels, _ = self._assign(x_t, centroids_t)
        return labels.cpu().numpy()

    def fit_predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Fit the model and return cluster assignments for the fit data."""
        self.fit(x)
        assert self.labels_ is not None
        return self.labels_

    def transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return Euclidean (not squared) distances from each point to every centroid."""
        self._check_fitted()
        x_t = self._as_tensor(x)
        centroids_t = torch.from_numpy(self.centroids_).to(
            device=x_t.device, dtype=x_t.dtype
        )
        return torch.cdist(x_t, centroids_t).cpu().numpy()

    def score_inertia(self, x: np.ndarray | torch.Tensor) -> float:
        """Sum of squared distances from each point in ``x`` to its nearest centroid.

        Note: ``inertia_`` holds the training-set value; use this method for val/test
        sets. (Named ``score_inertia`` rather than ``inertia`` so it doesn't shadow the
        attribute.)
        """
        self._check_fitted()
        x_t = self._as_tensor(x)
        centroids_t = torch.from_numpy(self.centroids_).to(
            device=x_t.device, dtype=x_t.dtype
        )
        _, sq = self._assign(x_t, centroids_t)
        return float(sq.sum().item())

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Persist the fitted model via joblib. Mirrors :class:`RecipeRegressor`."""
        self._check_fitted()
        output_path = ensure_parent_dir(path)
        payload: dict[str, Any] = {
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "n_init": self.n_init,
            "init_method": self.init_method,
            "random_state": self.random_state,
            "centroids_": self.centroids_,
            "inertia_": self.inertia_,
            "n_iter_": self.n_iter_,
        }
        joblib.dump(payload, output_path)

    @classmethod
    def load(cls, path: str | Path) -> KMeans:
        """Load a persisted KMeans model. Runs on CPU by default; call
        ``model.to_device("cuda")`` afterwards if GPU inference is desired."""
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or "centroids_" not in payload:
            raise ValueError(f"Artifact at {path} is not a recognized KMeans payload.")

        model = cls(
            n_clusters=int(payload.get("n_clusters", 8)),
            max_iter=int(payload.get("max_iter", 100)),
            tol=float(payload.get("tol", 1e-4)),
            n_init=int(payload.get("n_init", 10)),
            init_method=str(payload.get("init_method", "k-means++")),
            random_state=int(payload.get("random_state", 42)),
            device="cpu",
        )
        model.centroids_ = np.asarray(payload["centroids_"])
        model.inertia_ = (
            float(payload["inertia_"]) if payload.get("inertia_") is not None else None
        )
        model.n_iter_ = (
            int(payload["n_iter_"]) if payload.get("n_iter_") is not None else None
        )
        return model

    def to_device(self, device: str | torch.device) -> KMeans:
        """Change the device used for future ``predict`` / ``transform`` calls."""
        self.device = _resolve_device(device)
        return self

    # ---------------------------------------------------------- internals

    def _as_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dtype != torch.float32:
            x = x.float()
        return x.to(self.device)

    def _check_fitted(self) -> None:
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


# ------------------------------------------------------------------- helpers


def elbow_inertia(
    x: np.ndarray | torch.Tensor,
    k_values: list[int] | range,
    **kmeans_kwargs: Any,
) -> list[tuple[int, float]]:
    """Run KMeans for each ``K`` in ``k_values`` and return ``(K, inertia)`` pairs
    for elbow/knee plots. Keyword arguments are forwarded to :class:`KMeans`."""
    results: list[tuple[int, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), **kmeans_kwargs).fit(x)
        assert model.inertia_ is not None
        results.append((int(k), model.inertia_))
    return results
