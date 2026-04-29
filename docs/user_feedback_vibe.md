# Relevance Feedback - Validated Implementation Guide
## Text-Search Rocchio Feedback for Dynamic Recommendations

This document has been checked against the current app state:

- Search page: `app/pages/1_Search.py`
- Recipe cards: `app/components/recipe_cards.py`
- Retrieval service: `src/recipe_discovery/retrieval/service.py`
- Retrieval tests: `tests/test_retrieval.py`

The relevance-feedback concept is valid for this repo, but the older outline
used stale APIs and paths. This guide updates the implementation plan to match
the current code.

---

## Current-State Verdict

| Area | Status | Required adjustment |
|---|---|---|
| Retrieval service | Mostly ready | Add public text-query encoding and a feedback search method. |
| Search page | Ready with changes | Add feedback session keys, reset logic, and a feedback callback. |
| Recipe card component | Ready with changes | Add optional feedback action without removing shopping-list behavior. |
| Image / combined search | Defer | Keep feedback text-search-only for the MVP. |
| Tests | Needs new coverage | Add synthetic-vector tests for feedback behavior. |

---

## Stale Items in the Original Plan

These items would not work as written:

- `pages/1_Search.py` should be `app/pages/1_Search.py`.
- `app/components/recipe_card.py` should be `app/components/recipe_cards.py`.
- `RetrievalRequest(query_text=...)` is invalid. The dataclass uses `query`.
- `current_results` / `current_query` are not the page's active keys. The page
  uses `search_results_df`, `last_query`, and `last_search_mode`.
- `_search_candidates_for_vector(...)` was called with the wrong argument order.
  Current signature is:

```python
def _search_candidates_for_vector(
    self,
    request: RetrievalRequest,
    *,
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    limit_to_top_k: bool,
) -> pd.DataFrame:
    ...
```

- `encoder.encode(..., show_progress_bar=False)` is wrong for the local encoder
  usage. Current code uses `show_progress=False`.
- The current card renderer already supports shopping-list actions. Feedback
  must be additive and must not replace `on_add_to_shopping_list`.

---

## MVP Scope

Implement feedback only for successful text searches where
`last_search_mode == "text"`.

Do not show feedback controls for:

- Landing-state popular picks
- Image-only results
- Image + text combined results

This keeps the first pass easy to validate because the query vector and recipe
vectors both live in the SBERT embedding space.

---

## Desired Behavior

When a user clicks `Not relevant` on a text-search result:

1. The recipe ID is added to `st.session_state["feedback_excluded_ids"]`.
2. The stored original text query vector is adjusted away from all excluded
   recipe vectors.
3. Retrieval reruns with the adjusted vector and the active filters.
4. Excluded recipes are removed from the returned DataFrame.
5. The normal result display rerenders from `search_results_df`.

The adjustment is:

```text
adjusted = normalize(query_vec - alpha * mean(negative_vectors))
```

Recommended initial `alpha`: `0.3`.

---

## Step 1 - Retrieval Service

### Add `encode_text_query`

Add this public method to `RetrievalService`:

```python
def encode_text_query(self, query: str) -> np.ndarray:
    """Encode a text query into the loaded SBERT embedding space."""
    if self.encoder is None:
        raise RuntimeError("RetrievalService is not loaded.")
    if not query or not query.strip():
        raise ValueError("Search query is required for text feedback.")

    encoded = self.encoder.encode([query], show_progress=False)
    vec = np.asarray(encoded, dtype=float)[0]
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm
```

### Add `search_with_negative_feedback`

Add a text-first implementation. Keep `embedding_space` in the signature for a
future SigLIP extension, but only support `"text"` for this MVP.

```python
from dataclasses import replace

def search_with_negative_feedback(
    self,
    request: RetrievalRequest,
    *,
    query_vec: np.ndarray,
    negative_recipe_ids: set[str],
    alpha: float = 0.3,
    embedding_space: str = "text",
) -> pd.DataFrame:
    """Run text retrieval after applying negative Rocchio feedback."""
    if self.metadata is None or self.embeddings is None:
        raise RuntimeError("RetrievalService is not loaded.")
    if embedding_space != "text":
        raise ValueError("Only text feedback is supported in the MVP.")

    excluded = {str(recipe_id).strip() for recipe_id in negative_recipe_ids if str(recipe_id).strip()}
    if not excluded:
        return self.search(request)

    metadata_ids = self._normalize_recipe_ids(self.metadata["recipe_id"])
    excluded_series = pd.Series(list(excluded), dtype=str)
    excluded = set(self._normalize_recipe_ids(excluded_series).tolist())

    negative_mask = metadata_ids.isin(excluded).to_numpy()
    if not negative_mask.any():
        return self.search(request)

    query_vec = np.asarray(query_vec, dtype=float)
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    negative_vectors = self.embeddings[negative_mask]
    mean_negative = negative_vectors.mean(axis=0)
    adjusted = query_vec - alpha * mean_negative
    adjusted_norm = np.linalg.norm(adjusted)
    if adjusted_norm > 0:
        adjusted = adjusted / adjusted_norm
    else:
        adjusted = query_vec

    original_top_k = request.top_k
    feedback_request = replace(
        request,
        top_k=max(original_top_k + len(excluded) + 10, original_top_k),
    )
    candidates = self._search_candidates_for_vector(
        feedback_request,
        query_vec=adjusted,
        embeddings=self.embeddings,
        limit_to_top_k=False,
    )

    if not candidates.empty:
        candidate_ids = self._normalize_recipe_ids(candidates["recipe_id"])
        candidates = candidates.loc[~candidate_ids.isin(excluded)].copy()

    results = candidates.head(original_top_k).reset_index(drop=True)
    return self._attach_foodcom_images(results)
```

Why this works with the current service:

- It reuses the existing cosine-ranking and filtering logic in
  `_search_candidates_for_vector`.
- It keeps output format identical to `search(...)`: a `pd.DataFrame` with
  `similarity_score` and image URLs.
- It avoids introducing another result type for the Streamlit page.

---

## Step 2 - Search Page Session State

Add feedback keys to `_initialize_session_state()` in `app/pages/1_Search.py`:

```python
defaults = {
    ...
    "feedback_query_vec": None,
    "feedback_excluded_ids": set(),
    "feedback_active_request": None,
}
```

Add a helper:

```python
def _reset_feedback() -> None:
    st.session_state["feedback_query_vec"] = None
    st.session_state["feedback_excluded_ids"] = set()
    st.session_state["feedback_active_request"] = None
```

Call `_reset_feedback()` at the start of every new search or history-triggered
search. This prevents dislikes from one query polluting the next query.

---

## Step 3 - Store the Text Query Vector

In the existing text-search branch of `app/pages/1_Search.py`, store the query
vector and original request:

```python
elif has_query:
    _reset_feedback()
    st.session_state["feedback_query_vec"] = svc.encode_text_query(query)
    st.session_state["feedback_active_request"] = request
    results = svc.search(request)
    search_mode = "text"
```

For image-only and image + text branches, call `_reset_feedback()` and do not
store a feedback vector.

---

## Step 4 - Feedback Callback

Add this callback in `app/pages/1_Search.py`:

```python
def _on_negative_feedback(recipe_id: str) -> None:
    svc = get_retrieval_service()
    excluded = st.session_state["feedback_excluded_ids"]
    excluded.add(str(recipe_id))

    query_vec = st.session_state.get("feedback_query_vec")
    request = st.session_state.get("feedback_active_request")
    if query_vec is None or request is None:
        current = st.session_state.get("search_results_df")
        if isinstance(current, pd.DataFrame) and "recipe_id" in current.columns:
            keep = current["recipe_id"].astype(str) != str(recipe_id)
            st.session_state["search_results_df"] = current.loc[keep].reset_index(drop=True)
        return

    results = svc.search_with_negative_feedback(
        request,
        query_vec=query_vec,
        negative_recipe_ids=excluded,
        alpha=0.3,
    )
    st.session_state["search_results_df"] = results.copy()
```

The fallback branch is intentional. It lets the clicked card disappear even if
there is no stored vector, while avoiding a broken rerank.

---

## Step 5 - Recipe Card UI

Extend the existing `render_recipe_card` signature in
`app/components/recipe_cards.py` without removing shopping-list support:

```python
def render_recipe_card(
    recipe: Mapping[str, object],
    rank: int | None = None,
    *,
    display_mode: str = "Detailed",
    on_add_to_shopping_list: Callable[[Mapping[str, object]], None] | None = None,
    on_negative_feedback: Callable[[str], None] | None = None,
    feedback_key: str | None = None,
    widget_key_prefix: str = "search",
) -> None:
    ...
```

Inside the action-button area, add a feedback action only when provided:

```python
recipe_id = recipe.get("recipe_id") or recipe.get("id")
show_feedback = on_negative_feedback is not None and feedback_key and recipe_id is not None

if food_url or on_add_to_shopping_list is not None or show_feedback:
    num_actions = (
        int(bool(food_url))
        + int(on_add_to_shopping_list is not None)
        + int(show_feedback)
    )
    action_cols = st.columns(num_actions)
    ...
    if show_feedback:
        with action_cols[col_idx]:
            if st.button("Not relevant", key=feedback_key, use_container_width=True):
                on_negative_feedback(str(recipe_id))
                st.rerun()
```

Do not filter skipped cards inside `render_recipe_card`. The search page should
own the DataFrame state and exclusions. That keeps the card component reusable
for landing cards, result cards, and shopping-list-related actions.

---

## Step 6 - Wire Feedback Into Result Rendering

In the result loop in `app/pages/1_Search.py`, pass feedback props only for text
search results:

```python
show_feedback = search_mode == "text" and st.session_state.get("feedback_query_vec") is not None

for rank, (_, row) in enumerate(display_df.iterrows(), start=1):
    row_dict = row.to_dict()
    row_dict["_active_tags"] = get_active_tags(
        row_dict,
        tag_columns,
        max_tags=max_tags,
    )
    recipe_id = str(row_dict.get("recipe_id") or row_dict.get("id") or rank)
    render_recipe_card(
        row_dict,
        rank=rank,
        display_mode=display_mode,
        on_add_to_shopping_list=_add_recipe_to_shopping_list,
        on_negative_feedback=_on_negative_feedback if show_feedback else None,
        feedback_key=f"feedback_{recipe_id}_{rank}" if show_feedback else None,
        widget_key_prefix=f"results_{search_mode or 'text'}",
    )
```

Landing cards should not pass feedback props.

---

## Step 7 - Tests

Add focused service tests. These can live in `tests/test_retrieval.py` or a new
`tests/test_feedback.py`.

Suggested synthetic fixture:

```python
@pytest.fixture
def feedback_service() -> RetrievalService:
    class DummyEncoder:
        def encode(self, texts: list[str], *, show_progress: bool = False) -> np.ndarray:
            return np.array([[1.0, 0.0] for _ in texts], dtype=float)

    service = RetrievalService()
    service.encoder = DummyEncoder()
    service.embeddings = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=float,
    )
    service.metadata = pd.DataFrame(
        {
            "recipe_id": ["1", "2", "3", "4"],
            "name": ["A", "B", "C", "D"],
            "minutes": [10, 20, 30, 40],
            "n_ingredients": [3, 4, 5, 6],
        }
    )
    service.one_hot_tag_columns = []
    service._attach_foodcom_images = lambda df: df
    return service
```

Suggested tests:

```python
def test_encode_text_query_returns_normalized_vector(feedback_service):
    vec = feedback_service.encode_text_query("quick dinner")
    assert vec.shape == (2,)
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_feedback_excludes_negative_ids(feedback_service):
    request = RetrievalRequest(query="quick dinner", top_k=2)
    query_vec = feedback_service.encode_text_query("quick dinner")

    results = feedback_service.search_with_negative_feedback(
        request,
        query_vec=query_vec,
        negative_recipe_ids={"1"},
    )

    assert "1" not in results["recipe_id"].astype(str).tolist()
    assert len(results) <= 2


def test_feedback_rejects_non_text_embedding_space(feedback_service):
    request = RetrievalRequest(query="quick dinner", top_k=2)
    query_vec = feedback_service.encode_text_query("quick dinner")

    with pytest.raises(ValueError, match="Only text feedback"):
        feedback_service.search_with_negative_feedback(
            request,
            query_vec=query_vec,
            negative_recipe_ids={"1"},
            embedding_space="siglip",
        )
```

Run:

```bash
pytest tests/test_retrieval.py tests/test_recipe_cards.py tests/test_search_page_defaults.py
```

---

## Acceptance Criteria

| Criterion | Verification |
|---|---|
| `Not relevant` appears only on text-search results | Manual QA and search-mode condition. |
| Clicking feedback removes the card | Session DataFrame updates. |
| Results rerank using adjusted vectors | Service tests around excluded IDs and result shifts. |
| Feedback resets on new search | `_reset_feedback()` in every search branch. |
| Landing and image results have no feedback button | No feedback props passed in those paths. |
| Shopping-list action still works | Existing card tests and manual QA. |

---

## Implementation Order

| # | Task | File |
|---|---|---|
| 1 | Add `encode_text_query()` | `src/recipe_discovery/retrieval/service.py` |
| 2 | Add `search_with_negative_feedback()` | `src/recipe_discovery/retrieval/service.py` |
| 3 | Add synthetic service tests | `tests/test_retrieval.py` or `tests/test_feedback.py` |
| 4 | Add feedback session keys and `_reset_feedback()` | `app/pages/1_Search.py` |
| 5 | Store text query vector in text-search branch | `app/pages/1_Search.py` |
| 6 | Add optional feedback action to card renderer | `app/components/recipe_cards.py` |
| 7 | Add `_on_negative_feedback()` and wire it into result cards | `app/pages/1_Search.py` |
| 8 | Manual QA with 2-3 dislikes on a text query | Streamlit app |

Start with the service and tests. The UI wiring should only happen after the
service method has proven it returns the same DataFrame shape as `search()`.
