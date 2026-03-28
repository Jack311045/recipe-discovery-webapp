"""Utilities for train/validation/test splits and cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import pandas as pd


@dataclass
class SplitConfig:
    """Configuration for basic dataset splitting."""

    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10
    random_state: int = 42


def build_splits(
    df: pd.DataFrame, config: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test splits."""
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    shuffled = df.sample(frac=1.0, random_state=config.random_state).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * config.train_frac)
    val_end = train_end + int(n * config.val_frac)

    train_df = shuffled.iloc[:train_end].reset_index(drop=True)
    val_df = shuffled.iloc[train_end:val_end].reset_index(drop=True)
    test_df = shuffled.iloc[val_end:].reset_index(drop=True)
    return train_df, val_df, test_df


def kfold_indices(df: pd.DataFrame, n_splits: int = 5) -> Iterator[tuple[list[int], list[int]]]:
    """Yield train/test indices for simple contiguous k-fold cross-validation."""
    n = len(df)
    if n_splits <= 1 or n == 0:
        return

    fold_size = max(1, n // n_splits)
    all_indices = list(range(n))

    for fold in range(n_splits):
        start = fold * fold_size
        end = n if fold == n_splits - 1 else min(n, start + fold_size)
        test_idx = all_indices[start:end]
        train_idx = all_indices[:start] + all_indices[end:]
        yield train_idx, test_idx
