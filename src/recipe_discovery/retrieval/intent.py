"""Intent-aware query parsing and reranking helpers for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "want",
    "need",
    "recipe",
    "recipes",
    "food",
    "dish",
    "dishes",
}

_LOW_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "calories": (
        "low calorie",
        "low cal",
        "light",
        "weight loss",
        "lose weight",
        "diet",
        "healthy",
        "fat people",
        "overweight",
    ),
    "total fat": ("low fat", "low-fat", "lean", "heart healthy", "healthy"),
    "sugar": ("low sugar", "sugar free", "sugar-free", "diabetic", "healthy"),
    "sodium": ("low sodium", "low salt", "heart healthy", "healthy"),
    "carbohydrates": ("low carb", "low-carb", "keto", "ketogenic"),
}

_HIGH_TARGET_PATTERNS: dict[str, tuple[str, ...]] = {
    "protein": ("high protein", "protein rich", "protein-rich", "protein packed"),
    "calories": (
        "high calorie",
        "high-calorie",
        "indulgent",
        "decadent",
        "fattening",
    ),
    "total fat": ("high fat", "high-fat", "rich", "fried", "buttery"),
}

_QUICK_PATTERNS = (
    "quick",
    "fast",
    "easy",
    "weeknight",
    "simple",
    "in a hurry",
    "30 minutes",
    "15 minutes",
)

_SLOW_PATTERNS = (
    "slow cooked",
    "slow-cooked",
    "slow cooker",
    "long cook",
    "all day",
    "braised",
)

_LOW_SIGNAL_EXCLUDE_TERMS: dict[str, tuple[str, ...]] = {
    "calories": ("high calorie", "decadent", "fattening"),
    "total fat": ("high fat", "fried", "deep fried", "buttery"),
    "sugar": ("sweetened", "sugary", "candy"),
    "sodium": ("extra salty", "salted"),
    "carbohydrates": ("carb heavy", "starchy"),
}


@dataclass
class QueryIntent:
    """Structured representation of query intent signals."""

    original_query: str
    rewritten_query: str
    normalized_query: str
    query_terms: tuple[str, ...] = ()
    preferred_tags: tuple[str, ...] = ()
    avoid_tags: tuple[str, ...] = ()
    nutrition_targets: dict[str, str] = field(default_factory=dict)
    speed_target: str | None = None
    include_terms: tuple[str, ...] = ()
    exclude_terms: tuple[str, ...] = ()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalize_text(phrase)
    if not normalized_phrase:
        return False
    return bool(re.search(rf"\b{re.escape(normalized_phrase)}\b", text))


def _unique_in_order(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _tokenize_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9]+", text)
    return [term for term in terms if len(term) > 1 and term not in _STOPWORDS]


def _tag_phrase_variants(tag: str) -> tuple[str, ...]:
    phrase = _normalize_text(tag.replace("-", " "))
    if not phrase:
        return ()
    variants = [phrase]
    if phrase.endswith("s") and len(phrase) > 2:
        variants.append(phrase[:-1])
    elif len(phrase) > 2:
        variants.append(f"{phrase}s")
    return tuple(_unique_in_order(variants))


def _match_query_tags(query_norm: str, available_tags: Sequence[str]) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    avoid: list[str] = []

    for tag in available_tags:
        variants = _tag_phrase_variants(tag)
        if not variants:
            continue

        is_avoid = False
        for variant in variants:
            if any(
                _contains_phrase(query_norm, f"{negation} {variant}")
                for negation in ("no", "without", "not", "avoid")
            ):
                avoid.append(tag)
                is_avoid = True
                break

        if is_avoid:
            continue

        if any(_contains_phrase(query_norm, variant) for variant in variants):
            preferred.append(tag)

    preferred = _unique_in_order(preferred)
    avoid = _unique_in_order(avoid)
    preferred = [tag for tag in preferred if tag not in set(avoid)]
    return preferred, avoid


def _detect_nutrition_targets(query_norm: str) -> dict[str, str]:
    targets: dict[str, str] = {}

    for column, patterns in _LOW_TARGET_PATTERNS.items():
        if any(_contains_phrase(query_norm, pattern) for pattern in patterns):
            targets[column] = "low"

    for column, patterns in _HIGH_TARGET_PATTERNS.items():
        if any(_contains_phrase(query_norm, pattern) for pattern in patterns):
            targets[column] = "high"

    if _contains_phrase(query_norm, "healthy"):
        targets.setdefault("calories", "low")
        targets.setdefault("total fat", "low")
        targets.setdefault("sugar", "low")
        targets.setdefault("sodium", "low")
        targets.setdefault("protein", "high")

    return targets


def _detect_speed_target(query_norm: str) -> str | None:
    if any(_contains_phrase(query_norm, phrase) for phrase in _QUICK_PATTERNS):
        return "quick"
    if any(_contains_phrase(query_norm, phrase) for phrase in _SLOW_PATTERNS):
        return "slow"
    return None


def _build_rewritten_query(
    original_query: str,
    preferred_tags: Sequence[str],
    nutrition_targets: dict[str, str],
    speed_target: str | None,
) -> str:
    parts: list[str] = [original_query.strip()]

    if preferred_tags:
        parts.append(" ".join(tag.replace("-", " ") for tag in preferred_tags[:4]))

    if speed_target == "quick":
        parts.append("quick easy 30 minutes or less")
    elif speed_target == "slow":
        parts.append("slow cooked comfort meal")

    for nutrient, direction in nutrition_targets.items():
        parts.append(f"{direction} {nutrient}")

    rewritten = " ".join(part for part in parts if part).strip()
    return re.sub(r"\s+", " ", rewritten)


def understand_query_intent(query: str, *, available_tags: Sequence[str]) -> QueryIntent:
    """Parse a raw query into structured recipe-domain intent signals."""
    original = str(query or "").strip()
    query_norm = _normalize_text(original)
    if not query_norm:
        return QueryIntent(
            original_query=original,
            rewritten_query=original,
            normalized_query=query_norm,
        )

    preferred_tags, avoid_tags = _match_query_tags(query_norm, available_tags)
    nutrition_targets = _detect_nutrition_targets(query_norm)
    speed_target = _detect_speed_target(query_norm)

    query_terms = _tokenize_terms(query_norm)
    for tag in preferred_tags:
        query_terms.extend(_tokenize_terms(_normalize_text(tag.replace("-", " "))))
    query_terms = _unique_in_order(query_terms)

    include_terms = list(query_terms)
    exclude_terms: list[str] = []
    for nutrient, direction in nutrition_targets.items():
        if direction == "low":
            exclude_terms.extend(_LOW_SIGNAL_EXCLUDE_TERMS.get(nutrient, ()))

    rewritten_query = _build_rewritten_query(
        original,
        preferred_tags,
        nutrition_targets,
        speed_target,
    )

    return QueryIntent(
        original_query=original,
        rewritten_query=rewritten_query,
        normalized_query=query_norm,
        query_terms=tuple(query_terms),
        preferred_tags=tuple(preferred_tags),
        avoid_tags=tuple(avoid_tags),
        nutrition_targets=nutrition_targets,
        speed_target=speed_target,
        include_terms=tuple(_unique_in_order(include_terms)),
        exclude_terms=tuple(_unique_in_order(exclude_terms)),
    )


def _normalize_min_max(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    fill_value = float(values.median()) if values.notna().any() else 0.0
    dense = values.fillna(fill_value)
    min_value = float(dense.min())
    max_value = float(dense.max())
    if np.isclose(min_value, max_value):
        return pd.Series(np.zeros(len(dense), dtype=float), index=dense.index)
    return (dense - min_value) / (max_value - min_value)


def _row_lexical_score(row: pd.Series, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0

    name_tokens = set(_tokenize_terms(_normalize_text(str(row.get("name", "")))))
    ingredient_tokens = set(_tokenize_terms(_normalize_text(str(row.get("ingredients", "")))))
    body_tokens = set(
        _tokenize_terms(
            _normalize_text(
                " ".join(
                    [
                        str(row.get("description", "")),
                        str(row.get("steps", "")),
                    ]
                )
            )
        )
    )

    title_score = len(query_terms & name_tokens) / len(query_terms)
    ingredient_score = len(query_terms & ingredient_tokens) / len(query_terms)
    body_score = len(query_terms & body_tokens) / len(query_terms)
    return 0.55 * title_score + 0.3 * ingredient_score + 0.15 * body_score


def _nutrition_alignment(
    df: pd.DataFrame,
    nutrition_targets: dict[str, str],
) -> tuple[pd.Series, pd.Series]:
    if not nutrition_targets:
        zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return zeros, zeros

    alignment_parts: list[pd.Series] = []
    penalty = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    for column, target in nutrition_targets.items():
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() < 2:
            continue

        filled = values.fillna(values.median())
        percentile = filled.rank(method="average", pct=True)
        if target == "low":
            alignment = 1.0 - percentile
            penalty = penalty + (percentile - 0.85).clip(lower=0) * 1.5
        else:
            alignment = percentile
            penalty = penalty + (0.15 - percentile).clip(lower=0) * 1.5
        alignment_parts.append(alignment)

    if not alignment_parts:
        zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return zeros, penalty

    alignment_df = pd.concat(alignment_parts, axis=1)
    return alignment_df.mean(axis=1), penalty


def _speed_alignment(df: pd.DataFrame, speed_target: str | None) -> tuple[pd.Series, pd.Series]:
    zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if speed_target is None or "minutes" not in df.columns:
        return zeros, zeros

    minutes = pd.to_numeric(df["minutes"], errors="coerce")
    if minutes.notna().sum() < 2:
        return zeros, zeros

    filled = minutes.fillna(minutes.median())
    percentile = filled.rank(method="average", pct=True)

    if speed_target == "quick":
        score = 1.0 - percentile
        penalty = (percentile - 0.8).clip(lower=0) * 1.2
    else:
        score = percentile
        penalty = (0.2 - percentile).clip(lower=0) * 1.2
    return score, penalty


def _tag_alignment(
    df: pd.DataFrame,
    *,
    preferred_tags: Sequence[str],
    avoid_tags: Sequence[str],
) -> tuple[pd.Series, pd.Series]:
    zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)

    present_preferred = [tag for tag in preferred_tags if tag in df.columns]
    present_avoid = [tag for tag in avoid_tags if tag in df.columns]

    if present_preferred:
        preferred_matrix = (
            df[present_preferred]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(float)
            .clip(lower=0, upper=1)
        )
        preferred_score = preferred_matrix.mean(axis=1)
        # Penalize candidates that miss explicitly requested tags.
        preferred_missing_penalty = (1.0 - preferred_score) * 0.55
    else:
        preferred_score = zeros
        preferred_missing_penalty = zeros

    if present_avoid:
        avoid_penalty = (
            df[present_avoid]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(float)
            .clip(lower=0, upper=1)
            .mean(axis=1)
            * 0.45
        )
    else:
        avoid_penalty = zeros

    return preferred_score, avoid_penalty + preferred_missing_penalty


def _term_penalty(df: pd.DataFrame, exclude_terms: Sequence[str]) -> pd.Series:
    zeros = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    if not exclude_terms:
        return zeros

    text_series = (
        df.get("name", "").astype(str)
        + " "
        + df.get("description", "").astype(str)
        + " "
        + df.get("ingredients", "").astype(str)
    ).map(_normalize_text)

    penalties: list[float] = []
    for text in text_series.tolist():
        matches = sum(1 for term in exclude_terms if _contains_phrase(text, term))
        penalties.append((matches / max(len(exclude_terms), 1)) * 0.4)
    return pd.Series(penalties, index=df.index, dtype=float)


def rerank_candidates_with_intent(
    candidates: pd.DataFrame,
    *,
    intent: QueryIntent | None,
    one_hot_tag_columns: Sequence[str],
    base_score_column: str = "similarity_score",
) -> pd.DataFrame:
    """Apply interpretable, intent-aware reranking on top of dense candidates."""
    if candidates.empty:
        return candidates.copy()

    result = candidates.copy()
    if base_score_column not in result.columns:
        base_score_column = "similarity_score"

    result["base_ranking_score"] = _normalize_min_max(result[base_score_column])

    if intent is None or not intent.original_query.strip():
        result["intent_alignment_score"] = result["base_ranking_score"]
        return result.sort_values(
            ["intent_alignment_score", base_score_column],
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)

    query_terms = set(intent.query_terms)
    if query_terms:
        lexical_scores = [
            _row_lexical_score(row, query_terms)
            for _, row in result.iterrows()
        ]
        result["lexical_overlap_score"] = pd.Series(lexical_scores, index=result.index)
    else:
        result["lexical_overlap_score"] = 0.0

    tag_score, tag_penalty = _tag_alignment(
        result,
        preferred_tags=intent.preferred_tags,
        avoid_tags=intent.avoid_tags,
    )
    result["intent_tag_score"] = tag_score

    nutrition_score, nutrition_penalty = _nutrition_alignment(result, intent.nutrition_targets)
    result["intent_nutrition_score"] = nutrition_score

    speed_score, speed_penalty = _speed_alignment(result, intent.speed_target)
    result["intent_speed_score"] = speed_score

    term_penalty = _term_penalty(result, intent.exclude_terms)
    result["intent_contradiction_penalty"] = (
        tag_penalty + nutrition_penalty + speed_penalty + term_penalty
    )

    has_constraints = bool(
        intent.preferred_tags
        or intent.avoid_tags
        or intent.nutrition_targets
        or intent.speed_target
        or intent.exclude_terms
    )

    if has_constraints:
        result["intent_alignment_score"] = (
            0.5 * result["base_ranking_score"]
            + 0.2 * result["lexical_overlap_score"]
            + 0.17 * result["intent_tag_score"]
            + 0.09 * result["intent_nutrition_score"]
            + 0.04 * result["intent_speed_score"]
            - result["intent_contradiction_penalty"]
        )
    else:
        result["intent_alignment_score"] = (
            0.82 * result["base_ranking_score"]
            + 0.18 * result["lexical_overlap_score"]
        )

    sorted_result = result.sort_values(
        ["intent_alignment_score", base_score_column, "similarity_score"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)
    return sorted_result
