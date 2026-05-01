# Shopping List Smart Merge Implementation

## Summary

The shopping list now uses conservative ingredient canonicalization before
merging items. The goal is to reduce duplicate grocery rows without flattening
ingredients that should stay distinct for cooking. Each merged item still tracks
which recipes contributed to it and which original ingredient lines were added.

## Behavior

The merge layer keeps the existing shopping-list state fields:

1. `normalized_name`
2. `display_name`
3. `checked`
4. `category`
5. `source_recipes`

It also adds optional richer metadata:

1. `source_ingredients`: original ingredient lines grouped by recipe name
2. `merged_variants`: distinct ingredient labels that were merged into one item

Examples:

| Inputs | Shopping-list item | Notes |
|---|---|---|
| `salt and pepper`, then `salt` | `Salt and pepper` | Standalone salt merges only when the combined item already exists |
| `salt & pepper` | `Salt and pepper` | Ampersand normalizes to `and` |
| `salt and black pepper` | `Salt and pepper` | Ground/black pepper modifiers are normalized |
| `jasmine rice`, `basmati rice` | `Rice` | Conservative generic rice merge |
| `brown rice`, `rice vinegar`, `rice noodles` | Separate items | Culinary-specific rice products are preserved |
| `black pepper`, `bell pepper` | Separate items | Seasoning pepper and vegetable pepper are not merged |

## Data Flow

1. Recipe cards pass parsed ingredient lines and the recipe name into
   `add_ingredients_to_shopping_list()`.
2. `merge_ingredients()` cleans each ingredient line and calls
   `canonicalize_ingredient()`.
3. Canonicalization returns:
   - `canonical_key`
   - `display_name`
   - `category`
   - `merged_variant`
4. The shopping-list map is updated at the canonical key.
5. `source_recipes`, `source_ingredients`, and `merged_variants` are updated
   without duplicating repeated values.

## Category Presentation

Shopping items are grouped into grocery-style category sections:

1. Produce
2. Meat & Poultry
3. Seafood
4. Dairy & Eggs
5. Grains & Pasta
6. Baking
7. Spices & Seasonings
8. Condiments & Sauces
9. Pantry Staples
10. Other

The shopping-list page renders simple section headings with counts, such as
`Grains & Pasta (4)`. Recipe provenance stays compact but visually distinct:
each source recipe appears as its own small chip in the `From:` row.

## Backward Compatibility

Existing session-state items remain usable. Legacy category keys such as
`protein`, `dairy`, and `pantry` are mapped to the new aisle categories when the
shopping list is read.

Rows that do not yet have `source_ingredients` or `merged_variants` continue to
render normally. Those fields are added naturally as new ingredients are merged.

## Validation

Targeted tests cover:

1. Exact duplicate merge behavior
2. Simple plural merge behavior
3. Salt and pepper merging with source tracking
4. Conservative rice merging
5. Rice products that must stay separate
6. Black pepper versus bell pepper separation
7. `source_ingredients` provenance
8. Category inference

Run:

```bash
pytest tests/test_shopping_list.py
pytest tests/test_recipe_cards.py tests/test_search_ui.py
```
