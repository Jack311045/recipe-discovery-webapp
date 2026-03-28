"""Column contracts used by the pipeline."""

from __future__ import annotations

RECIPE_REQUIRED_COLUMNS = [
    "id",
    "name",
    "minutes",
    "tags",
    "nutrition",
    "steps",
    "ingredients",
    "description",
]

INTERACTION_REQUIRED_COLUMNS = [
    "user_id",
    "recipe_id",
    "date",
    "rating",
    "review",
]
