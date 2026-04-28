"""Auto-generate human-readable names for clusters.

Two complementary signals are used:

1. **TF-IDF distinctive words** — for each cluster, find words that are
   over-represented in cluster recipes relative to the full corpus. This
   produces data-driven labels like ``["chocolate", "vanilla", "butter"]``
   without needing any predefined vocabulary.

2. **Optional concept-vocabulary match** — if a list of food concepts is
   provided (the same vocabulary used by image search, for instance),
   we report which concepts overlap most with the distinctive words so
   the team can promote a cluster to a clean concept label
   (e.g. ``"desserts"`` or ``"asian stir-fries"``).

The TF-IDF path is always available; vocabulary matching is opt-in.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

# Common English stopwords + recipe-domain "filler" words. Hand-curated so we
# don't introduce a heavy NLP dependency for what is essentially a label task.
_DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles, pronouns, prepositions, conjunctions
        "a", "an", "the", "this", "that", "these", "those",
        "i", "you", "he", "she", "it", "we", "they", "my", "your", "our",
        "is", "are", "was", "were", "be", "been", "being", "am",
        "do", "does", "did", "have", "has", "had", "having",
        "of", "in", "on", "at", "by", "for", "with", "from", "to", "into",
        "and", "or", "but", "if", "as", "than", "so", "because", "while",
        "not", "no", "yes", "very", "just", "also", "only", "more", "most",
        "some", "any", "each", "every", "all", "both", "either", "neither",
        # Recipe-domain noise (process verbs, generic descriptors)
        "recipe", "recipes", "make", "makes", "making", "made",
        "use", "uses", "using", "used",
        "add", "adds", "added", "adding",
        "cook", "cooked", "cooking", "cooks",
        "easy", "quick", "simple", "best", "good", "great", "perfect",
        "delicious", "yummy", "tasty", "favorite", "favourite", "homemade",
        "minutes", "hour", "hours", "minute",
        "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
        "pound", "pounds", "ounce", "ounces", "lb", "oz", "tbsp", "tsp",
        "ingredients", "ingredient", "step", "steps", "instructions",
        "serve", "served", "serving", "servings",
        "place", "set", "let", "leave", "ready", "want", "need",
        "one", "two", "three", "four", "five", "small", "large", "medium",
        "really", "well", "way", "thing", "things", "time", "times",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z'-]+")


def _tokenize(text: str, stopwords: frozenset[str]) -> list[str]:
    """Lower-case, strip punctuation, drop stopwords and short tokens."""
    if not isinstance(text, str):
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 2 and t not in stopwords]


def _build_text_series(df: pd.DataFrame, text_columns: list[str]) -> pd.Series:
    """Concatenate selected text columns row-wise into a single string per recipe."""
    available = [c for c in text_columns if c in df.columns]
    if not available:
        raise ValueError(
            f"None of {text_columns} found in DataFrame columns: {list(df.columns)}"
        )
    return df[available].fillna("").agg(" ".join, axis=1)


def _document_frequencies(
    docs: Iterable[list[str]], n_docs: int
) -> tuple[Counter[str], Counter[str]]:
    """Return (term frequency over all docs, document frequency) counters."""
    tf: Counter[str] = Counter()
    df_count: Counter[str] = Counter()
    _ = n_docs  # accepted for symmetry; not used directly here
    for tokens in docs:
        tf.update(tokens)
        df_count.update(set(tokens))
    return tf, df_count


def distinctive_words_per_cluster(
    df: pd.DataFrame,
    *,
    cluster_column: str = "cluster",
    text_columns: tuple[str, ...] = ("name", "description", "tags"),
    top_n: int = 10,
    min_doc_freq: int = 5,
    stopwords: frozenset[str] | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """For each cluster, rank its words by how distinctive they are vs. the corpus.

    Scoring formula (essentially log-odds with smoothing): for cluster ``c`` and
    word ``w``::

        score(w, c) = freq(w | c) * log( (1 + freq(w | c)) / (1 + freq(w | not c)) )

    The first factor rewards in-cluster prevalence (so we don't surface rare
    typos); the log-ratio rewards relative over-representation. ``min_doc_freq``
    drops words that appear in too few recipes to be meaningful.

    Parameters
    ----------
    df : DataFrame containing at least ``cluster_column`` plus text fields.
    cluster_column : column holding integer cluster labels.
    text_columns : columns to concatenate as the per-recipe document.
    top_n : how many top words to return per cluster.
    min_doc_freq : minimum number of recipes a word must appear in (corpus-wide).
    stopwords : custom stopword set; defaults to a recipe-tuned list.

    Returns
    -------
    dict mapping cluster_id -> list of (word, score) sorted descending by score.
    """
    if cluster_column not in df.columns:
        raise ValueError(
            f"DataFrame missing cluster column {cluster_column!r}; "
            f"call attach_cluster_assignments first."
        )
    if stopwords is None:
        stopwords = _DEFAULT_STOPWORDS

    text_series = _build_text_series(df, list(text_columns))
    tokenized = [_tokenize(t, stopwords) for t in text_series]

    # Corpus-wide document frequency (for min_doc_freq filter)
    _, corpus_df = _document_frequencies(tokenized, n_docs=len(tokenized))

    cluster_ids = sorted(int(c) for c in df[cluster_column].unique())
    results: dict[int, list[tuple[str, float]]] = {}

    # Pre-compute per-cluster TF and rest-of-corpus TF to avoid quadratic work.
    cluster_membership = df[cluster_column].to_numpy()
    per_cluster_tf: dict[int, Counter[str]] = {}
    total_tf: Counter[str] = Counter()
    for tokens in tokenized:
        total_tf.update(tokens)
    for cid in cluster_ids:
        mask = cluster_membership == cid
        cluster_tokens: Counter[str] = Counter()
        for i in np.where(mask)[0]:
            cluster_tokens.update(tokenized[i])
        per_cluster_tf[cid] = cluster_tokens

    for cid in cluster_ids:
        in_tf = per_cluster_tf[cid]
        scored: list[tuple[str, float]] = []
        for word, count_in in in_tf.items():
            if corpus_df[word] < min_doc_freq:
                continue
            count_out = total_tf[word] - count_in
            # Log-odds with +1 smoothing to avoid log(0) and spurious rare-word spikes
            score = float(count_in) * np.log((1.0 + count_in) / (1.0 + count_out))
            scored.append((word, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        results[cid] = scored[:top_n]

    return results


def match_cluster_to_vocabulary(
    distinctive: dict[int, list[tuple[str, float]]],
    vocabulary: Iterable[str],
    *,
    top_n: int = 3,
) -> dict[int, list[str]]:
    """Map each cluster's distinctive words against a predefined concept list.

    For every concept word in ``vocabulary``, check whether it appears in (or is
    a substring of) the cluster's top distinctive words. Returns up to ``top_n``
    matched concepts per cluster, ordered by the rank of the supporting word.

    Useful when the team wants clean concept-level labels (``"pasta"``,
    ``"desserts"``) rather than raw distinctive tokens (``"chocolate"``,
    ``"butter"``).

    Parameters
    ----------
    distinctive : output of :func:`distinctive_words_per_cluster`.
    vocabulary : iterable of concept words. Multi-word phrases like
        ``"stir fry"`` are also supported (matched as a substring).
    top_n : maximum number of concepts to return per cluster.
    """
    vocab_list = [v.strip().lower() for v in vocabulary if v and v.strip()]
    results: dict[int, list[str]] = {}
    for cid, ranked_words in distinctive.items():
        matched: list[str] = []
        seen: set[str] = set()
        for word, _score in ranked_words:
            for concept in vocab_list:
                if concept in seen:
                    continue
                # Single-token concept: exact match. Multi-token: substring match
                # against the distinctive word's column-joined neighborhood.
                if " " in concept:
                    # Multi-token concepts can't be matched against single tokens;
                    # caller is expected to pass single concept words here.
                    continue
                if concept == word or (
                    len(concept) > 3 and (concept in word or word in concept)
                ):
                    matched.append(concept)
                    seen.add(concept)
                    if len(matched) >= top_n:
                        break
            if len(matched) >= top_n:
                break
        results[cid] = matched
    return results


def compose_cluster_name(
    distinctive_words: list[tuple[str, float]],
    matched_concepts: list[str] | None = None,
    *,
    max_words: int = 3,
) -> str:
    """Compose a short human-readable name from distinctive words and concepts.

    Strategy:
    - If concept matches exist, lead with the top concept and append distinctive
      words as modifiers (``"desserts: chocolate, vanilla"``).
    - Otherwise, fall back to the top ``max_words`` distinctive words joined by
      commas (``"chocolate, butter, sugar"``).
    """
    top_words = [w for w, _ in distinctive_words[:max_words]]
    if not top_words:
        return "(empty cluster)"
    if matched_concepts:
        head = matched_concepts[0]
        # Avoid repeating the head if it's already in top_words
        modifiers = [w for w in top_words if w != head][: max(0, max_words - 1)]
        if modifiers:
            return f"{head}: {', '.join(modifiers)}"
        return head
    return ", ".join(top_words)


def name_clusters(
    df: pd.DataFrame,
    *,
    cluster_column: str = "cluster",
    text_columns: tuple[str, ...] = ("name", "description", "tags"),
    vocabulary: Iterable[str] | None = None,
    top_n: int = 8,
    min_doc_freq: int = 5,
) -> dict[int, dict[str, object]]:
    """End-to-end convenience: distinctive words + optional vocabulary match
    + composed name, returned per cluster.

    Returns
    -------
    dict mapping cluster_id -> {"name": str, "distinctive": [(word, score), ...],
                                "matched_concepts": [str, ...]}
    """
    distinctive = distinctive_words_per_cluster(
        df,
        cluster_column=cluster_column,
        text_columns=text_columns,
        top_n=top_n,
        min_doc_freq=min_doc_freq,
    )
    matched_map: dict[int, list[str]] = {}
    if vocabulary is not None:
        matched_map = match_cluster_to_vocabulary(distinctive, vocabulary, top_n=3)

    out: dict[int, dict[str, object]] = {}
    for cid, words in distinctive.items():
        matched = matched_map.get(cid, [])
        out[cid] = {
            "name": compose_cluster_name(words, matched),
            "distinctive": words,
            "matched_concepts": matched,
        }
    return out


def summarize_cluster(df: pd.DataFrame, cluster_id: int) -> dict[str, object]:
    """Return a lightweight cluster summary (preserved for backward compatibility).

    For richer per-cluster information including auto-generated names use
    :func:`name_clusters`.
    """
    subset = df[df["cluster"] == cluster_id].copy()
    return {
        "cluster_id": cluster_id,
        "size": int(len(subset)),
    }
