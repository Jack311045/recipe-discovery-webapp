"""Retrieval evaluation helpers."""

from __future__ import annotations


def precision_at_k(relevant_flags: list[int], k: int) -> float:
    """Compute precision at k from binary relevance labels."""
    subset = relevant_flags[:k]
    if not subset:
        return 0.0
    return sum(subset) / len(subset)
