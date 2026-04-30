"""Helpers for rendering recipe results."""

from __future__ import annotations

import ast
import html
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping

import streamlit as st

try:
    from recipe_discovery.retrieval.image_fetcher import build_food_com_url as _build_food_com_url_impl
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    def _build_food_com_url_impl(recipe_id: object, recipe_name: str | None = None) -> str | None:
        return None


def _normalize_text_items(values: Iterable[object]) -> list[str]:
    """Normalize nested iterables into a flat list of readable strings."""
    items: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            items.extend(_normalize_text_items(value))
            continue
        text = re.sub(r"\s+", " ", str(value).strip())
        if text and text.lower() not in {"none", "null", "nan"}:
            items.append(text)
    return items


def _parse_serialized_list(raw_value: object, *, allow_comma_split: bool) -> list[str]:
    """Parse a recipe field that may contain list-like serialized text."""
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []
    if isinstance(raw_value, (list, tuple, set)):
        return _normalize_text_items(raw_value)

    text = str(raw_value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return []

    if text.startswith("[") != text.endswith("]"):
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, list):
                return _normalize_text_items(parsed_json)
        except json.JSONDecodeError:
            pass

        try:
            parsed_literal = ast.literal_eval(text)
            if isinstance(parsed_literal, (list, tuple, set)):
                return _normalize_text_items(parsed_literal)
        except (ValueError, SyntaxError):
            return []

    for delimiter in ("|", "\n"):
        if delimiter in text:
            return _normalize_text_items(text.split(delimiter))

    if allow_comma_split and "," in text:
        comma_parts = _normalize_text_items(text.split(","))
        if len(comma_parts) > 1:
            return comma_parts

    return _normalize_text_items([text])


def parse_ingredients(raw_value: object) -> list[str]:
    """Parse ingredients into bullet-ready list items."""
    return _parse_serialized_list(raw_value, allow_comma_split=True)


def parse_steps(raw_value: object) -> list[str]:
    """Parse preparation steps into numbered-list items."""
    return _parse_serialized_list(raw_value, allow_comma_split=False)


def parse_reviews(raw_value: object) -> list[str]:
    """Parse review text into readable review entries."""
    return _parse_serialized_list(raw_value, allow_comma_split=False)


def _as_float(value: object) -> float | None:
    """Convert arbitrary values to float, returning None on failure."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _as_int(value: object) -> int | None:
    """Convert arbitrary values to int, returning None on failure."""
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(round(numeric))


def _readable_tag(tag: str) -> str:
    """Format one-hot tag names for compact chip display."""
    cleaned = tag.replace("-", " ").replace("_", " ").strip()
    return cleaned.title() if cleaned else tag


def _render_tag_chips(tags: list[str]) -> None:
    """Render tag labels as lightweight chip-style pills."""
    if not tags:
        return

    chips = "".join(
        f"<span style='display:inline-block;padding:0.18rem 0.55rem;"
        f"margin:0.15rem 0.3rem 0.15rem 0;border-radius:999px;"
        f"background:#f4e8d1;color:#5a4738;border:1px solid #c8b38a;font-size:0.75rem;'>"
        f"{html.escape(_readable_tag(tag))}</span>"
        for tag in tags
    )
    st.markdown(chips, unsafe_allow_html=True)


def _get_description(recipe: Mapping[str, object]) -> str:
    raw = recipe.get("description", "")
    text = str(raw).strip()
    return "" if text.lower() in {"", "none", "nan", "null"} else text


def _collect_nutrition_values(recipe: Mapping[str, object]) -> list[tuple[str, str]]:
    """Collect nutrition values if present in the result row."""
    spec = [
        ("protein", "Protein", "g"),
        ("carbohydrates", "Carbs", "g"),
        ("total fat", "Total Fat", "g"),
        ("saturated fat", "Saturated Fat", "g"),
        ("sugar", "Sugar", "g"),
        ("sodium", "Sodium", "mg"),
        ("calories", "Calories", "kcal"),
    ]

    values: list[tuple[str, str]] = []
    for key, label, unit in spec:
        numeric = _as_float(recipe.get(key))
        if numeric is None:
            continue
        if unit == "kcal":
            values.append((label, f"{numeric:.0f} {unit}"))
        elif unit == "mg":
            values.append((label, f"{numeric:.0f} {unit}"))
        else:
            values.append((label, f"{numeric:.1f} {unit}"))
    return values


def _render_overview(
    recipe: Mapping[str, object],
    *,
    ingredient_items: list[str],
    step_items: list[str],
    clip_description: int,
) -> None:
    """Render overview content shared across card modes."""
    description = _get_description(recipe)
    if description:
        if len(description) > clip_description:
            st.write(f"{description[:clip_description]}…")
        else:
            st.write(description)

    rating = _as_float(recipe.get("rating"))
    ratings_count = _as_int(recipe.get("num_ratings"))
    overview_bits: list[str] = []
    if rating is not None:
        overview_bits.append(f"⭐ {rating:.2f}")
    if ratings_count is not None:
        overview_bits.append(f"🗳️ {ratings_count:,} ratings")
    if ingredient_items:
        overview_bits.append(f"🧂 {len(ingredient_items)} parsed ingredients")
    if step_items:
        overview_bits.append(f"📋 {len(step_items)} parsed steps")
    if overview_bits:
        st.caption("  ·  ".join(overview_bits))

    active_tags = recipe.get("_active_tags")
    if isinstance(active_tags, list) and active_tags:
        _render_tag_chips([str(tag) for tag in active_tags])

    x_proj = _as_float(recipe.get("x_proj"))
    y_proj = _as_float(recipe.get("y_proj"))
    if x_proj is not None and y_proj is not None:
        st.caption(f"📍 PCA coords: ({x_proj:.3f}, {y_proj:.3f})")


def _render_ingredients_section(items: list[str]) -> None:
    """Render ingredient bullets with graceful empty fallback."""
    if items:
        st.markdown("\n".join(f"- {item}" for item in items))
        return
    st.caption("Ingredients are unavailable for this recipe.")


def _render_steps_section(items: list[str]) -> None:
    """Render numbered recipe steps with graceful empty fallback."""
    if items:
        st.markdown("\n".join(f"{idx}. {item}" for idx, item in enumerate(items, start=1)))
        return
    st.caption("Preparation steps are unavailable for this recipe.")


def _render_reviews_section(items: list[str]) -> None:
    """Render review text with a centered empty state."""
    if not items:
        st.markdown(
            """
            <div class="menu-card-empty-reviews">No reviews</div>
            """,
            unsafe_allow_html=True,
        )
        return

    reviews = "".join(
        "<article class='menu-card-review'>"
        f"{html.escape(item)}"
        "</article>"
        for item in items
    )
    st.markdown(
        f"""
        <div class="menu-card-reviews">{reviews}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_nutrition_section(recipe: Mapping[str, object]) -> None:
    """Render nutrition metrics as a flex grid without truncation."""
    nutrition_values = _collect_nutrition_values(recipe)
    if not nutrition_values:
        st.caption("Nutrition details are unavailable for this recipe.")
        return

    cells = "".join(
        f"<div class='nutri-cell'>"
        f"<span class='nutri-label'>{html.escape(label)}</span>"
        f"<span class='nutri-value'>{html.escape(value)}</span>"
        f"</div>"
        for label, value in nutrition_values
    )
    st.markdown(
        f"""
        <style>
        .nutri-grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: space-between;
            margin: 0.3rem 0;
        }}
        .nutri-cell {{
            flex: 1 1 auto;
            min-width: 5.5rem;
            padding: 0.45rem 0.55rem;
            border: 1px solid rgba(200, 179, 138, 0.8);
            border-radius: 0.6rem;
            background: rgba(255, 250, 241, 0.92);
            text-align: center;
        }}
        .nutri-label {{
            display: block;
            font-size: 0.68rem;
            color: #6c5848;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.15rem;
            white-space: normal;
            word-wrap: break-word;
        }}
        .nutri-value {{
            display: block;
            font-size: 0.92rem;
            font-weight: 700;
            color: #3b2f2a;
            white-space: normal;
            word-wrap: break-word;
        }}
        </style>
        <div class="nutri-grid">{cells}</div>
        """,
        unsafe_allow_html=True,
    )


def _build_food_com_url(recipe: Mapping[str, object]) -> str | None:
    recipe_id = recipe.get("recipe_id") or recipe.get("id")
    if recipe_id is None:
        return None
    name = recipe.get("name")
    return _build_food_com_url_impl(recipe_id, str(name) if name is not None else None)


def _build_recipe_widget_suffix(recipe: Mapping[str, object], rank: int | None) -> str:
    """Build a stable, widget-safe suffix for per-recipe action keys."""
    raw = recipe.get("recipe_id") or recipe.get("id") or recipe.get("name") or "recipe"
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", str(raw)).strip("_")
    rank_part = str(rank) if rank is not None else "na"
    return f"{clean or 'recipe'}_{rank_part}"


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
    """Render a recipe result card with compact or detailed layouts."""
    title = str(recipe.get("name", recipe.get("title", "Untitled recipe")))
    heading_text = f"{rank}. {title}" if rank is not None else title
    image_url = recipe.get("image_url")
    score = _as_float(recipe.get("similarity_score"))
    minutes = _as_int(recipe.get("minutes"))
    n_ingredients = _as_int(recipe.get("n_ingredients"))
    n_steps = _as_int(recipe.get("n_steps"))
    ingredient_items = parse_ingredients(recipe.get("ingredients"))
    step_items = parse_steps(recipe.get("steps"))
    review_items = parse_reviews(recipe.get("all_reviews") or recipe.get("reviews"))
    compact = str(display_mode).strip().lower() == "compact"

    with st.container(border=True):
        image_text = str(image_url).strip() if image_url is not None else ""
        has_image = bool(image_text) and image_text.lower() not in {"none", "null", "nan"}

        meta_parts = []
        if minutes is not None:
            meta_parts.append(f"\u23f1 {int(minutes)} min")
        if n_ingredients is not None:
            meta_parts.append(f"\U0001f9c2 {int(n_ingredients)} ingredients")
        if n_steps is not None:
            meta_parts.append(f"\U0001f4cb {int(n_steps)} steps")

        signal_parts = [html.escape(part) for part in meta_parts]
        if score is not None:
            signal_parts.append(
                "<span class='menu-match-inline'>"
                f"Match {score:.2%}"
                "</span>"
            )

        header_cols = st.columns([1.05, 2.2], gap="medium")
        with header_cols[0]:
            if has_image:
                st.image(str(image_url), use_container_width=True)
            else:
                st.markdown(
                    "<div class='menu-card-image-placeholder'>No image</div>",
                    unsafe_allow_html=True,
                )

        with header_cols[1]:
            st.markdown(
                f"<h3 class='menu-card-title'>{html.escape(heading_text)}</h3>",
                unsafe_allow_html=True,
            )
            if signal_parts:
                st.markdown(
                    "<p class='menu-card-signal-row'>"
                    + "<span class='menu-card-meta-text'>"
                    + " &middot; ".join(signal_parts)
                    + "</span>"
                    + "</p>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div class='menu-card-divider'></div>", unsafe_allow_html=True)

        food_url = _build_food_com_url(recipe)
        recipe_id = recipe.get("recipe_id") or recipe.get("id")
        show_feedback = (
            on_negative_feedback is not None
            and feedback_key is not None
            and recipe_id is not None
        )

        # Count how many action buttons we have
        num_actions = (
            int(bool(food_url))
            + int(on_add_to_shopping_list is not None)
            + int(show_feedback)
        )

        if num_actions > 0:
            action_cols = st.columns(num_actions)
            col_idx = 0
            if food_url:
                with action_cols[col_idx]:
                    st.link_button("Website", food_url, use_container_width=True)
                col_idx += 1
            if on_add_to_shopping_list is not None:
                with action_cols[col_idx]:
                    suffix = _build_recipe_widget_suffix(recipe, rank)
                    add_clicked = st.button(
                        "🛒 Add to cart",
                        key=f"{widget_key_prefix}_add_to_list_{suffix}",
                        use_container_width=True,
                    )
                    if add_clicked:
                        on_add_to_shopping_list(recipe)
                col_idx += 1
            if show_feedback:
                with action_cols[col_idx]:
                    if st.button(
                        "✕ Not relevant",
                        key=feedback_key,
                        use_container_width=True,
                        help="Remove this result and adjust recommendations.",
                    ):
                        on_negative_feedback(str(recipe_id))
                        st.rerun()

        if compact:
            _render_overview(
                recipe,
                ingredient_items=ingredient_items,
                step_items=step_items,
                clip_description=240,
            )
            with st.expander("View ingredients"):
                _render_ingredients_section(ingredient_items)
            with st.expander("View steps"):
                _render_steps_section(step_items)
            return

        tabs = st.tabs(["Overview", "Nutrition", "Ingredients", "Steps", "Reviews"])
        with tabs[0]:
            _render_overview(
                recipe,
                ingredient_items=ingredient_items,
                step_items=step_items,
                clip_description=420,
            )
        with tabs[1]:
            _render_nutrition_section(recipe)
        with tabs[2]:
            _render_ingredients_section(ingredient_items)
        with tabs[3]:
            _render_steps_section(step_items)
        with tabs[4]:
            _render_reviews_section(review_items)
