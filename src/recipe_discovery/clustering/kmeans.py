"""From-scratch PyTorch k-means for recipe embedding clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from recipe_discovery.utils.io import ensure_parent_dir


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class KMeans:
    """K-means clustering with k-means++ init, n_init restarts, and empty-cluster reseeding.

    Parameters
    ----------
    n_clusters : int, default=8
    max_iter : int, default=100
    tol : float, default=1e-4
    n_init : int, default=10
        Number of random restarts; the lowest-inertia run is kept.
    init_method : {"k-means++", "random"}, default="k-means++"
    random_state : int, default=42
    device : str | torch.device | None, default=None
        ``None`` or ``"auto"`` selects CUDA when available, else CPU.
    verbose : bool, default=False

    Attributes (set after fit)
    --------------------------
    centroids_ : np.ndarray (n_clusters, n_features)
    labels_    : np.ndarray (n_samples,)
    inertia_   : float
    n_iter_    : int
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

    def _init_random(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        n = x.shape[0]
        idx = torch.randperm(n, generator=generator, device=x.device)[: self.n_clusters]
        return x[idx].clone()

    def _init_kmeans_pp(self, x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """k-means++ seeding: each centroid sampled with probability proportional to D(x)^2."""
        n, d = x.shape
        centroids = torch.empty(self.n_clusters, d, device=x.device, dtype=x.dtype)

        first_idx = torch.randint(0, n, (1,), generator=generator, device=x.device)
        centroids[0] = x[first_idx]
        closest_sq = torch.cdist(x, centroids[0:1]).squeeze(1) ** 2

        for k in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total <= 0:
                next_idx = torch.randint(0, n, (1,), generator=generator, device=x.device)
            else:
                probs = closest_sq / total
                next_idx = torch.multinomial(probs, 1, generator=generator)
            centroids[k] = x[next_idx]
            new_sq = torch.cdist(x, centroids[k : k + 1]).squeeze(1) ** 2
            closest_sq = torch.minimum(closest_sq, new_sq)

        return centroids

    @staticmethod
    def _assign(
        x: torch.Tensor, centroids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """E-step: assign each point to its nearest centroid."""
        dists_sq = torch.cdist(x, centroids) ** 2
        min_sq, labels = dists_sq.min(dim=1)
        return labels, min_sq

    def _update(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """M-step: update centroids; reseed empty clusters to farthest data points."""
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
                raise RuntimeError(
                    "All clusters empty after update; check that n_samples >= n_clusters."
                )
            dists_to_non_empty = torch.cdist(x, new_centroids[non_empty])
            min_dist, _ = dists_to_non_empty.min(dim=1)
            for k in empty_idx.tolist():
                farthest = int(min_dist.argmax().item())
                new_centroids[k] = x[farthest]
                min_dist[farthest] = -1.0

        return new_centroids

    def _run_once(
        self, x: torch.Tensor, run_seed: int
    ) -> tuple[torch.Tensor, torch.Tensor, float, int]:
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

    def fit(self, x: np.ndarray | torch.Tensor) -> KMeans:
        """Run k-means n_init times; keep the lowest-inertia result."""
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
        self.fit(x)
        assert self.labels_ is not None
        return self.labels_

    def transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return Euclidean distances from each point to every centroid."""
        self._check_fitted()
        x_t = self._as_tensor(x)
        centroids_t = torch.from_numpy(self.centroids_).to(
            device=x_t.device, dtype=x_t.dtype
        )
        return torch.cdist(x_t, centroids_t).cpu().numpy()

    def score_inertia(self, x: np.ndarray | torch.Tensor) -> float:
        """Sum of squared distances from points in x to their nearest centroid."""
        self._check_fitted()
        x_t = self._as_tensor(x)
        centroids_t = torch.from_numpy(self.centroids_).to(
            device=x_t.device, dtype=x_t.dtype
        )
        _, sq = self._assign(x_t, centroids_t)
        return float(sq.sum().item())

    def save(self, path: str | Path) -> None:
        """Persist the fitted model via joblib."""
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
        """Load a persisted KMeans model (CPU by default; call to_device for GPU)."""
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
        """Change the device used for future predict/transform calls."""
        self.device = _resolve_device(device)
        return self

    def _as_tensor(self, x: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dtype != torch.float32:
            x = x.float()
        return x.to(self.device)

    def _check_fitted(self) -> None:
        if self.centroids_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


def elbow_inertia(
    x: np.ndarray | torch.Tensor,
    k_values: list[int] | range,
    **kmeans_kwargs: Any,
) -> list[tuple[int, float]]:
    """Run KMeans for each K in k_values; return [(K, inertia), ...]."""
    results: list[tuple[int, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=int(k), **kmeans_kwargs).fit(x)
        assert model.inertia_ is not None
        results.append((int(k), model.inertia_))
    return results
