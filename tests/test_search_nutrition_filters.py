from pathlib import Path


SEARCH_PAGE_PATH = Path("app/pages/1_Search.py")


def test_search_page_exposes_calories_and_protein_sidebar_controls() -> None:
    source = SEARCH_PAGE_PATH.read_text(encoding="utf-8")

    assert "Apply calories requirement" in source
    assert "Maximum calories" in source
    assert "Apply protein requirement" in source
    assert "Minimum protein (g)" in source


def test_search_page_wires_nutrition_filters_into_request() -> None:
    source = SEARCH_PAGE_PATH.read_text(encoding="utf-8")

    assert "max_calories=max_calories" in source
    assert "min_protein=min_protein" in source
