# Usability Improvements - Validated Implementation Guide
## Landing State, Search History, Skeleton Loading

This proposal has been checked against the current app structure:

- Search page: `app/pages/1_Search.py`
- Search presentation helpers: `app/components/search_ui.py`
- Recipe card renderer: `app/components/recipe_cards.py`
- Runtime retrieval service: `src/recipe_discovery/retrieval/service.py`
- Service loader: `app/service_loader.py`

The three usability ideas are valid, but the original outline needed API corrections.
The app uses `RetrievalService`, `RetrievalRequest`, and `pandas.DataFrame`
results. It does not have an `engine.retrieve()` API or an `app/service.py`
module.

---

## Validation Summary

| Feature | Verdict | Notes |
|---|---|---|
| Landing State | Valid | Frontend/page-level only; use `get_retrieval_service().search(...)`. |
| Search History | Valid | Frontend/page-level only; use a keyed text input plus session state. |
| Skeleton Loading | Valid | Frontend/page-level only; use `st.empty()` around the current result region. |

Implementation should preserve the current retrieval behavior for ordinary
searches. New behavior should be additive and session-scoped.

---

## Current Search Flow

`app/pages/1_Search.py` currently:

1. Builds a `RetrievalRequest` from sidebar filters.
2. Routes to one of:
   - `svc.search(request)` for text search
   - `svc.search_by_image(image, request)` for image search
   - `svc.search_combined(query, image, request, alpha=alpha)` for combined search
3. Stores results in `st.session_state["search_results_df"]`.
4. Renders summary metrics, display controls, and cards with
   `render_recipe_card(...)`.

Any implementation should fit this shape rather than introducing a parallel
result format.

---

## Shared Session State

Add one initializer near the top of `app/pages/1_Search.py`, before widgets are
created:

```python
def _initialize_session_state() -> None:
    defaults = {
        "search_results_df": None,
        "last_query": "",
        "last_search_mode": "",
        "search_query_input": "",
        "search_history": [],
        "landing_results_df": None,
        "landing_query": "",
        "history_search_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
```

Notes:

- Use `search_query_input` as the `key` for the text input so history chips can
  repopulate it safely.
- Store DataFrames in session state, not `list[dict]`, to match existing code.

---

## Feature 1 - Landing State

### Goal

Avoid a blank first page by showing useful recipe results before the user runs a
search.

### Recommended Behavior

On the first page load, run one curated text query and cache the resulting
DataFrame for the session. Display those cards only while
`search_results_df is None`.

Suggested curated queries:

```python
LANDING_QUERIES = [
    "quick weeknight dinner",
    "healthy breakfast ideas",
    "easy comfort food",
    "simple vegetarian recipes",
]
```

Suggested helper:

```python
def _load_landing_results() -> pd.DataFrame:
    if st.session_state["landing_results_df"] is None:
        svc = get_retrieval_service()
        query = random.choice(LANDING_QUERIES)
        request = RetrievalRequest(
            query=query,
            top_k=8,
            max_time_minutes=45,
            max_ingredients=15,
            min_rating=4.0,
        )
        results = svc.search(request)
        if "rating" in results.columns:
            results = results.sort_values(
                by=["rating", "similarity_score"],
                ascending=[False, False],
                kind="mergesort",
            ).reset_index(drop=True)
        st.session_state["landing_results_df"] = results
        st.session_state["landing_query"] = query

    return st.session_state["landing_results_df"]
```

Render through the existing card path:

```python
def _render_landing_state() -> None:
    landing_placeholder = st.empty()
    if st.session_state["landing_results_df"] is None:
        with landing_placeholder.container():
            _render_skeleton_grid(8, label="Preparing popular picks...")

    landing_df = _load_landing_results()
    landing_placeholder.empty()

    st.markdown("### Popular starting points")
    st.caption(f"Showing highly rated matches for: {st.session_state['landing_query']}")

    tag_columns = infer_tag_columns(landing_df)
    for rank, (_, row) in enumerate(landing_df.iterrows(), start=1):
        row_dict = row.to_dict()
        row_dict["_active_tags"] = get_active_tags(row_dict, tag_columns, max_tags=8)
        render_recipe_card(row_dict, rank=rank, display_mode="Compact")
```

### Acceptance Criteria

- First load shows real recipe cards.
- First load shows skeleton cards while popular picks are loading.
- Landing results do not overwrite a user search.
- Landing search runs once per Streamlit session.
- Landing results should vary across sessions but prioritize highly rated recipes.

---

## Feature 2 - Search History

### Goal

Let users repeat recent text searches without retyping.

### Recommended Behavior

Store the last five unique non-empty text queries in session state. Render them
as compact buttons under the text input. Clicking a chip sets the text input and
reruns the page.

```python
def _add_to_history(query: str) -> None:
    query = query.strip()
    if not query:
        return
    history = [
        item for item in st.session_state["search_history"]
        if item.lower() != query.lower()
    ]
    st.session_state["search_history"] = [query, *history][:5]
```

```python
def _render_search_history() -> None:
    history = st.session_state["search_history"]
    if not history:
        return

    st.caption("Recent searches")
    cols = st.columns(min(len(history), 5))
    for idx, query in enumerate(history):
        label = query if len(query) <= 24 else f"{query[:21]}..."
        with cols[idx]:
            if st.button(label, key=f"history_query_{idx}", use_container_width=True):
                st.session_state["search_query_input"] = query
                st.rerun()
```

Update the text input:

```python
query = st.text_input(
    "Describe what you want to eat",
    placeholder="quick spicy tofu dinner...",
    key="search_query_input",
)
_render_search_history()
```

### Acceptance Criteria

- History appears after a successful text or combined search.
- Image-only searches do not add an empty history item.
- Duplicate queries move to the front instead of appearing twice.

---


## Feature 3 - Skeleton Loading Cards

### Goal

Keep the result region visually stable while retrieval runs.

### Recommended Behavior

Replace the full result-region blank/spinner with a placeholder container that
first renders skeleton cards, then renders real results after retrieval
finishes.

```python
def _render_skeleton_card() -> None:
    st.markdown(
        """
        <div class="result-skeleton">
          <div class="sk-img"></div>
          <div class="sk-body">
            <div class="sk-line sk-title"></div>
            <div class="sk-line sk-meta"></div>
            <div class="sk-line sk-short"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
```

Inject the CSS once near the page styles:

```css
.result-skeleton {
  display: grid;
  grid-template-columns: 260px 1fr;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  margin-bottom: 0.8rem;
}
.sk-img,
.sk-line {
  background: #e5e7eb;
  animation: pulse 1.2s ease-in-out infinite;
}
.sk-img {
  height: 150px;
  border-radius: 8px;
}
.sk-line {
  height: 14px;
  border-radius: 6px;
  margin-bottom: 0.75rem;
}
.sk-title { width: 70%; height: 22px; }
.sk-meta { width: 45%; }
.sk-short { width: 58%; }
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.45; }
}
```

Use it in the button handler:

```python
results_placeholder = st.empty()
with results_placeholder.container():
    for _ in range(top_k):
        _render_skeleton_card()

# Run svc.search / svc.search_by_image / svc.search_combined here.

results_placeholder.empty()
```

The existing lower-page render path can then display from
`st.session_state["search_results_df"]` as it does today.

### Acceptance Criteria

- The result area does not collapse during search.
- Skeleton count follows `top_k`.
- Existing summary metrics and display controls still render after results load.

---

## Implementation Order

| # | Change | Files | Risk |
|---|---|---|---|
| 1 | Session-state initializer | `app/pages/1_Search.py` | Low |
| 2 | Search history | `app/pages/1_Search.py` | Low |
| 3 | Landing state | `app/pages/1_Search.py` | Low |
| 4 | Skeleton loading | `app/pages/1_Search.py` | Low |


These changes are isolated UI improvements and should not change retrieval
semantics.

---

## Suggested Tests

- `tests/test_search_ui.py`
  - Add tests for history deduplication if helpers are moved into
    `app/components/search_ui.py`.
- `tests/test_search_page_defaults.py`
  - Keep source-level coverage for opt-in filters, display controls, and summary
    metrics.

---

## Final Recommendation

Proceed with landing state, search history, and skeleton loading. Relevance
feedback is intentionally deferred for a separate implementation pass.
