"""Tests for from-scratch PyTorch k-means.

Run with: ``pytest tests/test_kmeans.py -v``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from recipe_discovery.clustering.kmeans import KMeans, elbow_inertia

# ------------------------------------------------------------- fixtures


def _make_blobs(
    n_per_cluster: int = 50,
    centers: np.ndarray | None = None,
    std: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic well-separated Gaussian blobs for clustering tests."""
    if centers is None:
        centers = np.array([[0.0, 0.0], [5.0, 5.0], [0.0, 5.0], [5.0, 0.0]], dtype=np.float32)
    rng = np.random.default_rng(seed)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for i, c in enumerate(centers):
        x_parts.append(rng.normal(loc=c, scale=std, size=(n_per_cluster, c.shape[0])))
        y_parts.append(np.full(n_per_cluster, i))
    x = np.vstack(x_parts).astype(np.float32)
    y = np.concatenate(y_parts)
    return x, y


# ----------------------------------------------------- backward compatibility


def test_kmeans_fit_predict_returns_labels() -> None:
    """Preserves the original stub test so nothing downstream breaks."""
    x = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]], dtype=np.float32)
    model = KMeans(n_clusters=2, random_state=0)
    labels = model.fit_predict(x)
    assert len(labels) == len(x)


# ----------------------------------------------------- basic shape / API


def test_centroid_shape() -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x)
    assert model.centroids_ is not None
    assert model.centroids_.shape == (4, 2)


def test_labels_shape_and_range() -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x)
    assert model.labels_ is not None
    assert model.labels_.shape == (x.shape[0],)
    assert int(model.labels_.min()) >= 0
    assert int(model.labels_.max()) < 4


def test_predict_consistent_with_fit() -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x)
    assert np.array_equal(model.labels_, model.predict(x))


def test_fit_predict_equivalent_to_fit_then_labels() -> None:
    x, _ = _make_blobs()
    labels_via_fit_predict = KMeans(n_clusters=4, random_state=0, n_init=3).fit_predict(x)
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x)
    assert np.array_equal(labels_via_fit_predict, model.labels_)


def test_transform_shape_and_nonneg() -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x)
    distances = model.transform(x)
    assert distances.shape == (x.shape[0], 4)
    assert (distances >= 0).all()


def test_inertia_attribute_is_set() -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x)
    assert model.inertia_ is not None
    assert model.inertia_ > 0
    assert model.n_iter_ is not None
    assert 1 <= model.n_iter_ <= model.max_iter


# ----------------------------------------------------- correctness


def test_recovers_well_separated_blobs() -> None:
    """On well-separated Gaussian blobs the ARI with ground truth should exceed 0.9."""
    x, y_true = _make_blobs(seed=7)
    y_pred = KMeans(n_clusters=4, random_state=0, n_init=10).fit_predict(x)
    ari = adjusted_rand_score(y_true, y_pred)
    assert ari > 0.9, f"Expected ARI > 0.9, got {ari:.3f}"


def test_inertia_nonincreasing_within_single_run() -> None:
    """Lloyd's algorithm is monotone; inertia must not increase across iterations."""
    x, _ = _make_blobs()
    inertias: list[float] = []

    class _Hook(KMeans):
        @staticmethod
        def _assign(
            x_in: torch.Tensor, centroids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            labels, sq = KMeans._assign(x_in, centroids)
            inertias.append(float(sq.sum().item()))
            return labels, sq

    _Hook(n_clusters=4, random_state=0, n_init=1, max_iter=30).fit(x)
    diffs = np.diff(inertias)
    assert (diffs <= 1e-4).all(), (
        f"Inertia increased at step(s): {np.where(diffs > 1e-4)[0]}, diffs={diffs}"
    )


def test_more_restarts_improve_or_match_inertia() -> None:
    """More random restarts cannot hurt final inertia."""
    x, _ = _make_blobs()
    low_restart = KMeans(n_clusters=4, random_state=0, n_init=1, init_method="random").fit(x)
    many_restart = KMeans(n_clusters=4, random_state=0, n_init=10, init_method="random").fit(x)
    assert many_restart.inertia_ is not None and low_restart.inertia_ is not None
    assert many_restart.inertia_ <= low_restart.inertia_ + 1e-6


def test_kmeans_pp_matches_or_beats_random_on_average() -> None:
    """k-means++ mean inertia over 5 seeds should be <= random mean inertia."""
    x, _ = _make_blobs()
    random_inertias = [
        KMeans(n_clusters=4, random_state=s, n_init=1, init_method="random").fit(x).inertia_
        for s in range(5)
    ]
    pp_inertias = [
        KMeans(n_clusters=4, random_state=s, n_init=1, init_method="k-means++").fit(x).inertia_
        for s in range(5)
    ]
    assert float(np.mean(pp_inertias)) <= float(np.mean(random_inertias)) + 1e-6


def test_empty_cluster_reseeded() -> None:
    """With K > natural number of modes, no cluster should remain empty."""
    centers = np.array([[0.0, 0.0]], dtype=np.float32)
    x, _ = _make_blobs(n_per_cluster=20, centers=centers, std=0.01)
    model = KMeans(n_clusters=3, random_state=0, n_init=1).fit(x)
    unique = np.unique(model.labels_)
    assert len(unique) == 3


def test_score_inertia_on_separate_data() -> None:
    """score_inertia should evaluate fitted centroids on new data."""
    x_train, _ = _make_blobs(seed=1)
    x_test, _ = _make_blobs(seed=2)
    model = KMeans(n_clusters=4, random_state=0, n_init=5).fit(x_train)
    test_inertia = model.score_inertia(x_test)
    assert test_inertia >= 0
    assert model.inertia_ is not None
    assert test_inertia < 10 * model.inertia_  # same distribution, similar scale


def test_elbow_monotone_in_k() -> None:
    x, _ = _make_blobs()
    results = elbow_inertia(x, range(1, 6), random_state=0, n_init=3)
    ks = [k for k, _ in results]
    inertias = [inertia for _, inertia in results]
    assert ks == [1, 2, 3, 4, 5]
    for i in range(1, len(inertias)):
        assert inertias[i] <= inertias[i - 1] + 1e-6


def test_reproducibility_with_same_seed() -> None:
    """Same random_state on same device -> identical centroids."""
    x, _ = _make_blobs()
    a = KMeans(n_clusters=4, random_state=123, n_init=5).fit(x)
    b = KMeans(n_clusters=4, random_state=123, n_init=5).fit(x)
    assert np.allclose(a.centroids_, b.centroids_)
    assert a.inertia_ == pytest.approx(b.inertia_)


# ----------------------------------------------------- input handling


def test_accepts_torch_tensor() -> None:
    x, _ = _make_blobs()
    x_t = torch.from_numpy(x)
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x_t)
    assert model.centroids_ is not None
    assert model.centroids_.shape == (4, 2)


def test_accepts_float64_input() -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=3).fit(x.astype(np.float64))
    assert model.centroids_ is not None
    # Internal storage should be float32 for downstream consistency.
    assert model.centroids_.dtype == np.float32


# ----------------------------------------------------- persistence


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    x, _ = _make_blobs()
    model = KMeans(n_clusters=4, random_state=0, n_init=5).fit(x)
    artifact_path = tmp_path / "kmeans.joblib"
    model.save(artifact_path)
    assert artifact_path.exists()

    loaded = KMeans.load(artifact_path)
    assert np.allclose(loaded.centroids_, model.centroids_)
    assert loaded.inertia_ == pytest.approx(model.inertia_)
    assert loaded.n_clusters == model.n_clusters
    assert loaded.init_method == model.init_method
    # Predictions on new data should match bit-for-bit on CPU.
    assert np.array_equal(loaded.predict(x), model.predict(x))


def test_load_raises_on_corrupt_artifact(tmp_path: Path) -> None:
    import joblib

    bad_path = tmp_path / "not_kmeans.joblib"
    joblib.dump({"unrelated": "payload"}, bad_path)
    with pytest.raises(ValueError, match="recognized KMeans payload"):
        KMeans.load(bad_path)


# ----------------------------------------------------- error handling


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        KMeans(n_clusters=3).predict(np.zeros((5, 2), dtype=np.float32))


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        KMeans(n_clusters=3).transform(np.zeros((5, 2), dtype=np.float32))


def test_score_inertia_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        KMeans(n_clusters=3).score_inertia(np.zeros((5, 2), dtype=np.float32))


def test_invalid_n_clusters() -> None:
    with pytest.raises(ValueError, match="n_clusters"):
        KMeans(n_clusters=0)


def test_invalid_init_method() -> None:
    with pytest.raises(ValueError, match="init_method"):
        KMeans(n_clusters=3, init_method="fancy")


def test_invalid_max_iter() -> None:
    with pytest.raises(ValueError, match="max_iter"):
        KMeans(n_clusters=3, max_iter=0)


def test_invalid_n_init() -> None:
    with pytest.raises(ValueError, match="n_init"):
        KMeans(n_clusters=3, n_init=0)


def test_fewer_samples_than_clusters_raises() -> None:
    x = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="n_samples"):
        KMeans(n_clusters=5, random_state=0).fit(x)
