# Shopping Cart, Food.com Passthrough, and Feedback Button Breakdown

This document explains three user-facing systems in the recipe discovery app:

- The shopping cart / shopping list ingredient pipeline.
- The Food.com website passthrough and image lookup behavior.
- The "Not relevant" feedback button and how it adjusts search results.

It is written as a technical reference and as a basis for a presentation script.

## High-Level Story

The app does more than show semantic search results. Each recipe card becomes an action surface:

1. A user searches for recipes.
2. Each result card can open the original Food.com page.
3. Each result card can add its parsed ingredients into a persistent shopping list.
4. Each result card can be marked "Not relevant" so the retrieval service removes that recipe and shifts the query away from similar recipes.

The implementation is intentionally lightweight. Most state is held in Streamlit session state, while heavier search behavior stays in `RetrievalService`.

## Key Files

- `app/components/recipe_cards.py`
  - Renders recipe cards.
  - Builds Food.com website links.
  - Shows the add-to-shopping-list action.
  - Shows the "Not relevant" feedback action.

- `app/pages/1_Search.py`
  - Owns the search-page callbacks.
  - Connects recipe-card actions to shopping-list helpers and retrieval feedback.
  - Stores active query vectors, active requests, excluded recipe IDs, and cached results.

- `app/components/shopping_list.py`
  - Owns shopping-list state and merge logic.
  - Normalizes ingredients.
  - Canonicalizes smart merges.
  - Tracks recipe provenance.
  - Groups items by grocery-style categories.

- `app/pages/5_Shopping_List.py`
  - Renders the shopping list page.
  - Shows manual item entry, progress metrics, checkboxes, source chips, merge captions, and remove controls.

- `src/recipe_discovery/retrieval/image_fetcher.py`
  - Builds Food.com recipe URLs.
  - Fetches Food.com image URLs from JSON-LD when local image data is missing.

- `src/recipe_discovery/retrieval/service.py`
  - Attaches image URLs to results.
  - Runs text, image, combined, and negative-feedback retrieval.

## Shopping Cart Implementation

### User Flow

The shopping-list flow starts on a recipe card:

1. Search results are rendered by `_render_result_grid()` in `app/pages/1_Search.py`.
2. Each card is rendered by `render_recipe_card()` in `app/components/recipe_cards.py`.
3. If shopping-list support is available, the card shows a compact `+` button.
4. Clicking the button calls `_add_recipe_to_shopping_list(recipe)` in `app/pages/1_Search.py`.
5. The recipe's ingredient field is parsed with `parse_ingredients()`.
6. Parsed ingredients are passed to `add_ingredients_to_shopping_list()`.
7. The shopping-list helper merges them into `st.session_state["shopping_list_items"]`.
8. The app shows a toast / success message with added and merged counts.

### Recipe Card Entry Point

In `render_recipe_card()`, the add button is only shown when the caller provides an `on_add_to_shopping_list` callback.

The card does not know how the shopping list works. It only says: "Here is the recipe object; call this callback if the user clicks add."

This keeps the recipe card reusable. The card handles UI. The search page handles behavior.

### Ingredient Parsing

Ingredients can come from the dataset in several forms:

- Python-style list strings, such as `['flour', 'salt']`.
- JSON-style list strings, such as `["flour", "salt"]`.
- Pipe-separated text, such as `salt|pepper|garlic`.
- Comma-separated text, such as `salt, pepper, garlic`.
- Already parsed Python lists or tuples.

`parse_ingredients()` in `app/components/recipe_cards.py` normalizes these into a clean `list[str]`.

This matters because the shopping-list merge layer expects a sequence of ingredient lines, not raw serialized dataset text.

### Session State Model

Shopping-list state is stored in:

```python
st.session_state["shopping_list_items"]
```

It is a dictionary keyed by a normalized ingredient name. Each value stores display and provenance data.

Representative item shape:

```python
{
    "normalized_name": "rice",
    "display_name": "Rice",
    "checked": False,
    "category": "grains_pasta",
    "source_recipes": ["Recipe A", "Recipe B"],
    "source_ingredients": {
        "Recipe A": ["jasmine rice"],
        "Recipe B": ["basmati rice"]
    },
    "merged_variants": ["jasmine rice", "basmati rice"]
}
```

This structure lets the list combine practical grocery items while still preserving where each ingredient came from.

### Normalization Layer

The first step is `normalize_ingredient_name()`.

It:

- Converts text to lowercase.
- Trims whitespace.
- Converts `&` into `and`.
- Converts underscores and hyphens into spaces.
- Removes most punctuation.
- Collapses repeated whitespace.

Examples:

```text
"  Olive-oil!  " -> "olive oil"
"Salt & Pepper" -> "salt and pepper"
"Fresh   Basil" -> "fresh basil"
```

This gives stable dictionary keys while still preserving a separate human-readable `display_name`.

### Canonicalization Layer

The smart merge behavior is handled by `canonicalize_ingredient()`.

It returns a `CanonicalIngredient` object with:

- `canonical_key`: the key used in session state.
- `display_name`: the preferred label shown to the user.
- `category`: the inferred grocery category.
- `merged_variant`: the original ingredient wording to show later under "Combines".

This layer is deliberately conservative. It merges obvious grocery equivalents, but avoids risky culinary merges.

### Exact and Plural Merge

After canonicalization, `_resolve_existing_key()` handles exact matches and simple plural variants.

Examples:

```text
"onion" + "onion" -> one item
"onion" + "onions" -> one item
```

It only handles simple one-letter plural endings. That avoids aggressive stemming that could merge unrelated ingredients.

### Salt and Pepper Merge

The app recognizes combined seasoning phrases:

```text
"salt and pepper"
"salt & pepper"
"salt and black pepper"
"pepper and salt"
```

These canonicalize to:

```text
"salt and pepper"
```

There is one extra nuance. If `salt and pepper` already exists in the shopping list, later standalone `salt`, `black pepper`, `ground pepper`, or similar seasoning pepper can merge into that combined item.

But vegetable peppers do not merge with black pepper.

Examples kept separate:

```text
"bell pepper"
"red pepper"
"green pepper"
"poblano"
```

This distinction is important for trust. A grocery shopper expects black pepper and bell pepper to be completely different items.

### Rice Merge

The app also merges common interchangeable rice names into a practical grocery item:

```text
"jasmine rice"
"basmati rice"
"white rice"
"long grain rice"
```

These canonicalize to:

```text
"rice"
```

Specialized rice products stay separate:

```text
"brown rice"
"wild rice"
"arborio rice"
"sushi rice"
"sticky rice"
"rice flour"
"rice vinegar"
"rice noodles"
```

This keeps the merge useful without flattening culinary differences.

### Provenance Tracking

The shopping list does not just merge ingredients. It records where each item came from.

Two fields handle provenance:

- `source_recipes`: a unique list of recipe names.
- `source_ingredients`: a mapping from recipe name to the original ingredient lines.

Example:

```python
"source_recipes": ["Chicken Bowl", "Salmon Dinner"],
"source_ingredients": {
    "Chicken Bowl": ["jasmine rice"],
    "Salmon Dinner": ["basmati rice"]
}
```

This lets the UI show a compact "From:" line while still keeping the raw ingredient text available for explanation or future expansion.

### Merge Variants

`merged_variants` stores distinct terms that were combined.

Example:

```python
"merged_variants": ["jasmine rice", "basmati rice"]
```

The shopping list page displays this as:

```text
Combines: jasmine rice, basmati rice
```

This is useful for presentation because it makes the "smart merge" visible without overwhelming the main item label.

### Category Inference

`infer_item_category()` assigns a stable grocery category.

Current category keys include:

- `produce`
- `meat_poultry`
- `seafood`
- `dairy_eggs`
- `grains_pasta`
- `baking`
- `spices_seasonings`
- `condiments_sauces`
- `pantry_staples`
- `other`

Categories are inferred from keyword sets, plus special cases for pepper and rice. For example:

```text
"spinach" -> Produce
"chicken breast" -> Meat & Poultry
"rice" -> Grains & Pasta
"salt and pepper" -> Spices & Seasonings
"rice vinegar" -> Condiments & Sauces
"rice flour" -> Baking
```

### Shopping List Rendering

`app/pages/5_Shopping_List.py` renders the shopping list page.

It provides:

- Manual item entry.
- Item count metrics.
- Completed count.
- Remaining count.
- Progress bar.
- Remove checked.
- Clear all.
- Optional category grouping.
- Checkbox per item.
- Remove button per item.
- Source recipe chips.
- Merge variant captions.

Source recipe display is compact:

- Show up to 3 recipe names as chips.
- If there are more, show `+N more`.

This improves readability when one item came from many recipes.

### Shopping Cart Presentation Talk Track

Use this as a script:

> The shopping list is session-state based, so it follows the user as they search and move between pages. The recipe card does not directly manage the list. It exposes a callback, and the search page wires that callback to the shopping-list helper.
>
> When a recipe is added, we parse the serialized ingredient field into a clean list. Then each ingredient goes through normalization and conservative canonicalization. Normalization gives us stable keys like `olive oil`, while canonicalization handles a few high-value grocery cases, such as salt and pepper or common white rice varieties.
>
> The important design choice is that we merge only when the merge is obvious. Bell pepper does not merge with black pepper. Rice vinegar does not merge with rice. This avoids the kind of "smart" behavior that looks impressive but breaks trust.
>
> Every merged item keeps provenance. We store which recipes contributed to it, and we also keep the original ingredient line per recipe. That is why the shopping list can show `Rice` as one item while still saying it came from two recipes and combines `jasmine rice` and `basmati rice`.

## Food.com Website Passthrough

### User Flow

Each recipe card can include a `Website` button.

The flow is:

1. The recipe result includes a `recipe_id` and usually a recipe name.
2. `_build_food_com_url()` in `app/components/recipe_cards.py` asks the retrieval image helper to build the original Food.com URL.
3. `build_food_com_url()` in `src/recipe_discovery/retrieval/image_fetcher.py` creates the URL slug.
4. The recipe card renders `st.link_button("Website", food_url)`.
5. Clicking the button takes the user to the Food.com recipe page.

### URL Construction

The Food.com URL builder uses:

- Recipe name for the slug.
- Recipe ID for the unique page identifier.

Example transformation:

```text
Recipe name: "Creamy Mushroom Pasta"
Recipe ID: 12345
URL: https://www.food.com/recipe/creamy-mushroom-pasta-12345
```

The slug builder:

- Lowercases the name.
- Removes unsupported characters.
- Converts whitespace to hyphens.
- Collapses repeated hyphens.
- Uses `recipe` as the slug fallback if no name is available.

The recipe ID is normalized so values like `12345.0` become `12345`.

### Why This Is a Passthrough

The app does not try to recreate the entire Food.com page. It provides discovery, ranking, filtering, and shopping-list actions, then passes the user through to the original source for the canonical recipe page.

Benefits:

- Keeps the original recipe source accessible.
- Reduces the need to duplicate full website content.
- Gives users a familiar external page for final verification.
- Uses the dataset recipe ID to preserve source identity.

### Food.com Image Lookup

The project also uses Food.com as an image source when image URLs are missing or unusable.

This happens through `attach_foodcom_images()`:

1. Retrieval returns a result dataframe.
2. `RetrievalService` calls `_attach_foodcom_images(results)`.
3. The helper checks whether `image_url` is missing, empty, fallback, or placeholder-like.
4. For rows needing images, it builds the Food.com recipe URL.
5. It downloads the page with `requests`.
6. It parses the Food.com page with BeautifulSoup.
7. If the page says `Add your photo` and has no `Photo by` credit, it treats the recipe as genuinely missing a submitted image.
8. If the page has a `Photo by` credit, it tries structured JSON-LD, Open Graph / Twitter metadata, and recipe-like image tags.
9. It rejects placeholder URLs, logos, defaults, and SVG placeholders.
10. It falls back to a generic Unsplash food image if nothing reliable is found.

### Food.com Image Optimizations

The image lookup includes several guardrails:

- `@lru_cache(maxsize=512)` caches `fetch_food_com_image()` by recipe ID and name.
- `REQUEST_TIMEOUT = 4` prevents a slow external page from hanging too long.
- Missing-image lookups run through a small bounded worker pool, so a result page can resolve several Food.com photos without waiting for every request sequentially.
- When lookups are run serially, `POLITE_DELAY = 0.3` spaces out multiple requests.
- A browser-like user agent improves compatibility with Food.com.
- Placeholder detection avoids showing logos or default assets as recipe photos.
- `RetrievalService._attach_image_urls()` first loads `image_map.parquet` when available, so Food.com requests are only a fallback.

### Food.com Presentation Talk Track

Use this as a script:

> The Food.com integration has two parts. First, every recipe card can pass the user through to the original Food.com page. We build the URL from the Food.com recipe ID and a slugified recipe name, then render it as a Streamlit link button.
>
> Second, we use Food.com as a fallback image source. If a recipe does not already have a good image URL, we fetch the Food.com page, parse its JSON-LD recipe metadata, and extract the image. This is cached and timeout-limited, so repeated lookups are cheaper and bad pages do not block the app indefinitely.
>
> The key design principle is source preservation. The app helps with search, ranking, and list-building, but it keeps a clear route back to the original recipe.

## "Not Relevant" Feedback Button

### User Flow

The "Not relevant" button appears on recipe cards when the search page provides a negative-feedback callback.

The flow is:

1. User clicks `Not relevant` on a result card.
2. The recipe card calls `on_negative_feedback(recipe_id)`.
3. `_on_negative_feedback()` in `app/pages/1_Search.py` adds that ID to `feedback_excluded_ids`.
4. If there is an active query vector and request, the search page calls `RetrievalService.search_with_negative_feedback()`.
5. The retrieval service shifts the query vector away from the negative example.
6. Results are recomputed, excluded IDs are removed, and the page reruns.

### Feedback State

The search page stores feedback-specific session state:

```python
feedback_query_vec
feedback_excluded_ids
feedback_active_request
feedback_embedding_space
```

The meaning:

- `feedback_query_vec`: the original encoded query vector.
- `feedback_excluded_ids`: recipes the user marked as not relevant.
- `feedback_active_request`: the current retrieval request, including filters and top-k.
- `feedback_embedding_space`: either `text` or `siglip`.

This allows the feedback system to work across:

- Text search.
- Image search.
- Combined image plus text search.

### Query Vector Setup

When a search runs:

- Text search stores `svc.encode_text_query(query)`.
- Image search stores `svc.encode_image_query(image)`.
- Combined search stores `svc.encode_combined_query(query, image, alpha=alpha)`.

The embedding space is tracked because text search uses the SBERT-style text embedding matrix, while image and image-plus-text use SigLIP embeddings.

### Fallback Behavior

The feedback handler has graceful fallback cases:

- If the user is dismissing a landing-page card, it simply removes that card from the landing results.
- If a result dataframe exists but there is no query vector or request, it removes only the selected row.
- If full feedback state is available, it performs vector-based negative feedback.

This means the button remains useful even when the app cannot rerank.

### Rocchio-Style Negative Feedback

`search_with_negative_feedback()` uses a Rocchio-like update.

The core idea:

```python
adjusted_query = original_query - alpha * mean(negative_vectors)
```

In this project:

- `alpha` defaults to `0.3` from the search-page callback.
- The negative vectors are embeddings for the recipes marked not relevant.
- The adjusted query is normalized after subtraction.
- Retrieval then runs with the adjusted query vector.

In plain language:

> Move the query representation away from the recipes the user rejected.

This is more nuanced than simply removing one card. It tries to reduce similar unwanted results as well.

### Exclusion and Refill

After adjusting the query:

1. The service asks for a slightly larger candidate set.
2. It filters out all excluded recipe IDs.
3. It returns the top `original_top_k` recipes.
4. It attaches image URLs again.

The larger candidate set matters because if one result is removed, we need replacement candidates available. The service uses:

```python
original_top_k + len(excluded) + 10
```

as the temporary feedback request size.

### Interaction With Cached Results

The search page also maintains a hidden result pool for "Get a few more."

When feedback exclusions exist, `_filter_excluded_results()` removes excluded recipe IDs from the cached pool. That prevents a recipe marked "Not relevant" from coming back through the load-more cache.

### Feedback Presentation Talk Track

Use this as a script:

> The `Not relevant` button is not just a delete button. It has a delete fallback, but when we have an active query vector, it becomes vector feedback.
>
> For every search, we save the encoded query vector and the active request. If the user marks a recipe as not relevant, we store its recipe ID and look up its embedding. Then we move the query vector away from the mean of those rejected recipe vectors.
>
> This is a Rocchio-style negative feedback step. It means the next result set should not only remove the exact recipe, but should also reduce recipes that are close to the rejected example.
>
> The implementation supports text search and SigLIP-based image search because the page records which embedding space the query came from. That is why the same button can work across text, image, and combined search modes.

## End-to-End Demo Script

### Demo 1: Shopping List

1. Search for a recipe query, for example `chicken rice dinner`.
2. Click the `+` button on one recipe.
3. Open the shopping list page.
4. Point out item count, completed count, remaining count, and progress bar.
5. Add another recipe with overlapping ingredients.
6. Show that ingredients like rice merge.
7. Show source chips under `From:`.
8. Show the `Combines:` caption for merged variants.
9. Check an item and remove checked items.

Suggested narration:

> This is not just appending ingredient text. It is building a normalized grocery list. The app merges obvious duplicates, keeps the item readable, and preserves where it came from.

### Demo 2: Food.com Passthrough

1. On a recipe card, click `Website`.
2. Show that the user lands on the original Food.com page.
3. Return to the app.
4. Explain that image URLs also use Food.com when local image data is missing.

Suggested narration:

> The app is a discovery and workflow layer over Food.com data. It helps users search and plan, but the source page remains one click away.

### Demo 3: Not Relevant Feedback

1. Search for a broad query.
2. Identify a result that is off-intent.
3. Click `Not relevant`.
4. Show that the result disappears.
5. Explain that the app also adjusts the query vector away from that result.

Suggested narration:

> This creates a lightweight personalization loop within the session. The user can steer the search without typing a new query.

## Design Tradeoffs

### Why Session State

Streamlit session state is simple and appropriate for a prototype:

- No database required.
- State persists across page reruns.
- Search and shopping pages can share state.
- It is easy to reset, sanitize, and inspect.

Tradeoff:

- It is per-session, not permanent user storage.

### Why Conservative Ingredient Merging

Aggressive ingredient merging can create bad grocery lists.

The current implementation chooses high-confidence merges:

- Exact duplicates.
- Simple plurals.
- Salt and pepper variants.
- Common white rice variants.

It avoids ambiguous merges:

- Black pepper versus bell pepper.
- Rice versus rice vinegar.
- Rice versus rice flour.
- Rice versus specialty rice types.

Tradeoff:

- Some useful merges are not handled yet.
- But the merges that do happen are easier to trust.

### Why Food.com Fallback Images Are Cached

Fetching external pages can be slow and unreliable.

The implementation uses:

- Preloaded image maps when available.
- LRU cache for repeated Food.com lookups.
- Request timeout.
- Small delay between requests.
- Fallback image when extraction fails.

Tradeoff:

- First-time missing-image lookup can still be slow.
- But repeated lookup is much faster.

### Why Rocchio Feedback

Rocchio-style feedback is a good fit because the app already has embeddings.

It is:

- Simple.
- Explainable.
- Works with text and image embedding spaces.
- Does not require model retraining.

Tradeoff:

- It is session-local.
- It depends on embedding quality.
- It can only steer from examples the user marks.

## Future Upgrade Ideas

- Persist shopping lists to user accounts or local files.
- Add expandable per-recipe ingredient provenance under each merged item.
- Add quantity parsing and unit-aware merging.
- Add more canonicalization rules for oils, herbs, and dairy while keeping them conservative.
- Prefetch Food.com images in a background task rather than during retrieval.
- Add positive feedback, such as "more like this."
- Store multiple feedback events and expose a "reset feedback" action.
- Add explanation text after feedback, such as "Removed recipes similar to X."

## One-Minute Summary

The shopping list, Food.com passthrough, and feedback button turn search results into an interactive planning workflow.

The shopping list converts recipe ingredients into a normalized grocery list, merging obvious duplicates while preserving recipe provenance. The Food.com passthrough keeps the original recipe source accessible and uses Food.com structured data as a fallback image source. The "Not relevant" button gives users a lightweight way to steer search results by moving the query vector away from rejected recipes.

Together, these features make the app feel less like a static search demo and more like a usable recipe discovery assistant.
