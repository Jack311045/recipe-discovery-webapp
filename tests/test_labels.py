"""Tests for cluster auto-naming."""

from __future__ import annotations

import pandas as pd
import pytest

from recipe_discovery.clustering.labels import (
    compose_cluster_name,
    distinctive_words_per_cluster,
    match_cluster_to_vocabulary,
    name_clusters,
    summarize_cluster,
)

# ------------------------------------------------------------- fixtures


def _make_synthetic_clusters() -> pd.DataFrame:
    """Build a tiny corpus with three planted themes:

    Cluster 0 — desserts (chocolate, sugar, vanilla)
    Cluster 1 — pastas   (pasta, parmesan, tomato)
    Cluster 2 — salads   (salad, lettuce, vinaigrette)
    """
    rows: list[dict[str, object]] = []
    for _ in range(8):
        rows.append({
            "name": "chocolate cake",
            "description": "rich chocolate dessert with vanilla cream",
            "tags": "dessert chocolate vanilla sugar",
            "cluster": 0,
        })
    for _ in range(8):
        rows.append({
            "name": "spaghetti carbonara",
            "description": "italian pasta with parmesan and pancetta",
            "tags": "pasta italian parmesan tomato",
            "cluster": 1,
        })
    for _ in range(8):
        rows.append({
            "name": "caesar salad",
            "description": "crisp lettuce salad with vinaigrette dressing",
            "tags": "salad lettuce vinaigrette greens",
            "cluster": 2,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------- distinctive_words


def test_distinctive_words_recovers_planted_themes() -> None:
    df = _make_synthetic_clusters()
    result = distinctive_words_per_cluster(df, top_n=10, min_doc_freq=1)

    cluster_0_words = {w for w, _ in result[0]}
    cluster_1_words = {w for w, _ in result[1]}
    cluster_2_words = {w for w, _ in result[2]}

    assert "chocolate" in cluster_0_words
    assert "vanilla" in cluster_0_words
    assert "pasta" in cluster_1_words
    assert "parmesan" in cluster_1_words
    assert "salad" in cluster_2_words
    assert "lettuce" in cluster_2_words

    # No theme should leak into a sibling cluster's top words
    assert "chocolate" not in cluster_1_words
    assert "pasta" not in cluster_0_words
    assert "salad" not in cluster_0_words


def test_distinctive_words_respects_min_doc_freq() -> None:
    """Words appearing in fewer than min_doc_freq docs are filtered out."""
    df = _make_synthetic_clusters()
    # Add a unique typo that appears once -- should NOT surface
    df.loc[0, "description"] = df.loc[0, "description"] + " uniquetypoxyz"
    result = distinctive_words_per_cluster(df, top_n=20, min_doc_freq=2)
    cluster_0_words = {w for w, _ in result[0]}
    assert "uniquetypoxyz" not in cluster_0_words


def test_distinctive_words_respects_top_n() -> None:
    df = _make_synthetic_clusters()
    result = distinctive_words_per_cluster(df, top_n=3, min_doc_freq=1)
    for cid, words in result.items():
        assert len(words) <= 3, f"Cluster {cid} returned {len(words)} words, expected <= 3"


def test_distinctive_words_scores_are_descending() -> None:
    df = _make_synthetic_clusters()
    result = distinctive_words_per_cluster(df, top_n=10, min_doc_freq=1)
    for cid, words in result.items():
        scores = [s for _, s in words]
        assert scores == sorted(scores, reverse=True), (
            f"Cluster {cid} scores not descending: {scores}"
        )


def test_distinctive_words_filters_stopwords() -> None:
    """Hand-curated stopwords like 'recipe', 'easy', 'cup' should never appear."""
    df = _make_synthetic_clusters()
    # Sprinkle stopwords aggressively
    df["description"] = "easy recipe " + df["description"].astype(str) + " serve in cup"
    result = distinctive_words_per_cluster(df, top_n=20, min_doc_freq=1)
    forbidden = {"easy", "recipe", "serve", "cup"}
    for cid, words in result.items():
        present = {w for w, _ in words} & forbidden
        assert not present, f"Cluster {cid} surfaced stopwords: {present}"


def test_distinctive_words_missing_cluster_column_raises() -> None:
    df = pd.DataFrame({"name": ["a"], "description": ["b"], "tags": ["c"]})
    with pytest.raises(ValueError, match="cluster column"):
        distinctive_words_per_cluster(df)


def test_distinctive_words_missing_text_columns_raises() -> None:
    df = pd.DataFrame({"cluster": [0, 1]})
    with pytest.raises(ValueError, match="None of"):
        distinctive_words_per_cluster(df)


# ----------------------------------------------------- vocabulary match


def test_vocabulary_match_finds_concept_words() -> None:
    df = _make_synthetic_clusters()
    distinctive = distinctive_words_per_cluster(df, top_n=10, min_doc_freq=1)
    vocab = ["pasta", "salad", "dessert", "chocolate", "soup"]
    matched = match_cluster_to_vocabulary(distinctive, vocab, top_n=2)

    # Cluster 1 (pasta theme) should match "pasta"
    assert "pasta" in matched[1]
    # Cluster 2 (salad theme) should match "salad"
    assert "salad" in matched[2]
    # Cluster 0 (dessert theme) should match "chocolate" or "dessert"
    assert any(c in matched[0] for c in ("chocolate", "dessert"))
    # "soup" appears in nothing -> shouldn't show up anywhere
    for _cid, concepts in matched.items():
        assert "soup" not in concepts


def test_vocabulary_match_handles_empty_vocabulary() -> None:
    df = _make_synthetic_clusters()
    distinctive = distinctive_words_per_cluster(df, top_n=5, min_doc_freq=1)
    matched = match_cluster_to_vocabulary(distinctive, [])
    for concepts in matched.values():
        assert concepts == []


def test_vocabulary_match_respects_top_n() -> None:
    df = _make_synthetic_clusters()
    distinctive = distinctive_words_per_cluster(df, top_n=20, min_doc_freq=1)
    vocab = ["pasta", "italian", "tomato", "parmesan", "spaghetti", "carbonara"]
    matched = match_cluster_to_vocabulary(distinctive, vocab, top_n=2)
    for concepts in matched.values():
        assert len(concepts) <= 2


# ----------------------------------------------------- compose_cluster_name


def test_compose_name_with_concept_match() -> None:
    distinctive = [("chocolate", 5.0), ("vanilla", 3.0), ("sugar", 2.0)]
    matched = ["dessert"]
    name = compose_cluster_name(distinctive, matched)
    assert name.startswith("dessert")
    assert "chocolate" in name


def test_compose_name_without_concept_match_uses_distinctive() -> None:
    distinctive = [("chocolate", 5.0), ("vanilla", 3.0), ("sugar", 2.0)]
    name = compose_cluster_name(distinctive, matched_concepts=None)
    assert name == "chocolate, vanilla, sugar"


def test_compose_name_empty_distinctive() -> None:
    name = compose_cluster_name([], matched_concepts=None)
    assert name == "(empty cluster)"


def test_compose_name_concept_overlaps_distinctive() -> None:
    """Avoid duplicating: ``compose_cluster_name([("pasta", 1.0)], ["pasta"])``
    should not produce ``"pasta: pasta"``."""
    name = compose_cluster_name([("pasta", 5.0)], ["pasta"])
    assert name == "pasta"


# ----------------------------------------------------- end-to-end name_clusters


def test_name_clusters_end_to_end_with_vocabulary() -> None:
    df = _make_synthetic_clusters()
    vocab = ["pasta", "salad", "dessert"]
    result = name_clusters(df, vocabulary=vocab, top_n=5, min_doc_freq=1)

    assert set(result.keys()) == {0, 1, 2}
    for _cid, info in result.items():
        assert "name" in info and isinstance(info["name"], str)
        assert "distinctive" in info
        assert "matched_concepts" in info
        assert info["name"]  # non-empty

    # Names should reflect themes
    assert any(t in result[1]["name"] for t in ("pasta", "italian", "parmesan"))
    assert any(t in result[2]["name"] for t in ("salad", "lettuce"))


def test_name_clusters_works_without_vocabulary() -> None:
    df = _make_synthetic_clusters()
    result = name_clusters(df, vocabulary=None, top_n=5, min_doc_freq=1)
    for _cid, info in result.items():
        assert info["matched_concepts"] == []
        # Falls back to distinctive-words name
        assert info["name"] != "(empty cluster)"


# ----------------------------------------------------- backward compatibility


def test_summarize_cluster_preserved() -> None:
    """The original summarize_cluster API stays available."""
    df = _make_synthetic_clusters()
    summary = summarize_cluster(df, 0)
    assert summary["cluster_id"] == 0
    assert summary["size"] == 8
