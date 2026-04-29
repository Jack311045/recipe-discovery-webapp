"""Search page defaults should not enforce restrictive filters by default."""

from pathlib import Path


def test_search_page_uses_optional_time_and_ingredient_filters() -> None:
    source = Path("app/pages/1_Search.py").read_text(encoding="utf-8")

    assert 'use_time_limit = st.checkbox("Apply max cooking time filter", value=False)' in source
    assert (
        'use_ingredient_limit = st.checkbox("Apply max ingredients filter", value=False)'
        in source
    )


def test_search_request_uses_optional_filter_values() -> None:
    source = Path("app/pages/1_Search.py").read_text(encoding="utf-8")

    assert "max_time_minutes=max_time" in source
    assert "max_ingredients=max_ingredients" in source


def test_search_page_includes_display_controls() -> None:
    source = Path("app/pages/1_Search.py").read_text(encoding="utf-8")
    helper = Path("app/components/search_ui.py").read_text(encoding="utf-8")

    assert "Sort displayed results" in source
    assert "Best match" in helper
    assert "Highest rating" in helper
    assert "Fastest" in helper
    assert "Fewest ingredients" in helper
    assert "Card view" not in source
    assert "DISPLAY_MODES" not in helper


def test_search_page_includes_summary_metrics_section() -> None:
    source = Path("app/pages/1_Search.py").read_text(encoding="utf-8")

    assert "Results" in source
    assert "Avg cook time" in source
    assert "Avg rating" in source
    assert "Avg calories" in source


def test_recipe_cards_include_reviews_tab() -> None:
    source = Path("app/components/recipe_cards.py").read_text(encoding="utf-8")

    assert '"Reviews"' in source
    assert "all_reviews" in source
