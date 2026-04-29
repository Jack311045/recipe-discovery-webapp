"""Floating shopping-cart menu for search and discovery pages."""

from __future__ import annotations

import html
from collections.abc import Mapping

import streamlit as st
import streamlit.components.v1 as components

from app.components.shopping_list import (
    ensure_shopping_list_state,
    get_checked_item_count,
    get_shopping_items,
    get_shopping_list_count,
)


SHOPPING_LIST_PAGE = "pages/5_Shopping_List.py"
SHOPPING_LIST_SWITCH_LABEL = "Open full shopping list via Streamlit"


def _format_item_label(item: Mapping[str, object]) -> str:
    display_name = str(item.get("display_name") or item.get("normalized_name") or "").strip()
    if not display_name:
        return "Unnamed item"
    if len(display_name) <= 44:
        return display_name
    return f"{display_name[:41]}..."


def _format_sources(sources: object) -> str:
    if not isinstance(sources, list) or not sources:
        return ""
    source_strings = [str(source).strip() for source in sources if str(source).strip()]
    if not source_strings:
        return ""
    if len(source_strings) == 1:
        return source_strings[0]
    if len(source_strings) <= 3:
        return ", ".join(source_strings)
    return f"{', '.join(source_strings[:2])} (+{len(source_strings) - 2} more)"


def _render_cart_items(preview_items: list[dict[str, object]], total_items: int) -> str:
    if total_items == 0:
        return "<p class='floating-cart-empty'>Add ingredients from recipe cards.</p>"

    item_markup = ["<div class='floating-cart-list'>"]
    for item in preview_items:
        label = html.escape(_format_item_label(item))
        source = html.escape(_format_sources(item.get("source_recipes")))
        item_markup.append("<div class='floating-cart-item'>")
        item_markup.append(f"<div class='floating-cart-item-name'>{label}</div>")
        if source:
            item_markup.append(f"<div class='floating-cart-item-source'>From: {source}</div>")
        item_markup.append("</div>")
    item_markup.append("</div>")

    if total_items > len(preview_items):
        hidden_count = total_items - len(preview_items)
        item_markup.append(
            f"<p class='floating-cart-more'>+{hidden_count} more item(s) in the full list.</p>"
        )

    return "".join(item_markup)


def render_floating_shopping_cart() -> None:
    """Render a fixed top-right cart icon with a compact click-away menu.

    Uses the native HTML Popover API so clicking outside the cart panel closes
    it without needing another click on the cart button.
    """
    ensure_shopping_list_state()

    total_items = get_shopping_list_count()
    checked_items = get_checked_item_count()
    remaining_items = max(total_items - checked_items, 0)
    preview_items = get_shopping_items()[:6]
    item_markup = _render_cart_items(preview_items, total_items)

    # Badge only shown when count > 0
    badge_html = ""
    if total_items > 0:
        badge_html = (
            f"<span class='floating-cart-badge'>{total_items}</span>"
        )

    st.markdown(
        f"""
<style>
/* ---- Ancestor overflow fix ---- */
.stApp,
.stApp > header,
.stApp [data-testid="stHeader"],
.stApp > div,
.main,
.main > div,
.block-container {{
    overflow: visible !important;
}}

/* ---- Cart wrapper ---- */
.floating-cart-wrapper {{
    position: fixed;
    top: 0.55rem;
    right: 5.25rem;
    z-index: 999999;
    font-family: "Lora", "Merriweather", serif;
    pointer-events: auto;
}}

/* ---- Summary / trigger button ---- */
.floating-cart-trigger {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.3rem;
    min-width: 2.6rem;
    min-height: 2.15rem;
    padding: 0.18rem 0.62rem;
    border-radius: 999px;
    border: 1px solid rgba(90, 66, 47, 0.92);
    background: linear-gradient(180deg, #7a5a43 0%, #6B4F3A 100%);
    color: #fff9ef;
    box-shadow: 0 5px 16px rgba(68, 45, 29, 0.24);
    cursor: pointer;
    user-select: none;
    transition: transform 140ms ease, box-shadow 140ms ease;
}}

.floating-cart-trigger:hover {{
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(68, 45, 29, 0.30);
    filter: brightness(1.05);
}}

.floating-cart-icon {{
    color: #fff9ef;
    font-size: 1.12rem;
    line-height: 1;
}}

.floating-cart-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 1.2rem;
    height: 1.2rem;
    padding: 0 0.28rem;
    border-radius: 999px;
    background: #c0392b;
    color: #fff;
    font-size: 0.7rem;
    font-weight: 800;
    line-height: 1;
}}

/* ---- Dropdown panel ---- */
.floating-cart-panel {{
    position: fixed;
    top: 3.05rem;
    right: 5.25rem;
    left: auto;
    bottom: auto;
    margin: 0;
    width: min(20rem, calc(100vw - 1.5rem));
    padding: 0.8rem;
    border: 1px solid rgba(183, 159, 115, 0.92);
    border-radius: 0.85rem;
    background: rgba(255, 250, 241, 0.99);
    box-shadow: 0 14px 36px rgba(68, 45, 29, 0.22);
    backdrop-filter: blur(8px);
    animation: cartSlideIn 180ms ease-out;
    z-index: 999999;
}}

.floating-cart-panel::backdrop {{
    background: transparent;
}}

@keyframes cartSlideIn {{
    from {{ opacity: 0; transform: translateY(-6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

.floating-cart-title {{
    margin: 0 0 0.1rem 0;
    color: #5b4434;
    font-size: 1rem;
    font-weight: 800;
}}

.floating-cart-summary {{
    margin: 0 0 0.52rem 0;
    color: #6c5848;
    font-size: 0.78rem;
}}

.floating-cart-list {{
    display: grid;
    gap: 0.32rem;
    margin: 0.25rem 0 0.55rem 0;
    max-height: 16rem;
    overflow-y: auto;
}}

.floating-cart-item {{
    padding: 0.42rem 0.48rem;
    border: 1px solid rgba(200, 179, 138, 0.8);
    border-radius: 0.55rem;
    background: rgba(255, 253, 247, 0.9);
}}

.floating-cart-item-name {{
    color: #3b2f2a;
    font-size: 0.82rem;
    font-weight: 650;
    line-height: 1.25;
}}

.floating-cart-item-source,
.floating-cart-empty,
.floating-cart-more {{
    color: #746152;
    font-size: 0.72rem;
    line-height: 1.25;
}}

.floating-cart-item-source {{
    margin-top: 0.12rem;
}}

.floating-cart-link {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    min-height: 2.1rem;
    border-radius: 999px;
    border: 1px solid #5A422F;
    background: linear-gradient(180deg, #7a5a43 0%, #6B4F3A 100%);
    color: #fff9ef !important;
    font-size: 0.82rem;
    font-weight: 750;
    text-decoration: none !important;
    cursor: pointer;
    transition: transform 140ms ease, box-shadow 140ms ease;
}}

.floating-cart-link:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(68, 45, 29, 0.22);
}}

@media (max-width: 760px) {{
    .floating-cart-wrapper {{
        top: 0.45rem;
        right: 3.15rem;
    }}

    .floating-cart-panel {{
        top: 2.9rem;
        right: 0.5rem;
        width: min(18.5rem, calc(100vw - 1rem));
    }}
}}
</style>
<div class="floating-cart-wrapper">
<button
    class="floating-cart-trigger"
    type="button"
    popovertarget="floating-cart-panel"
    popovertargetaction="toggle"
    aria-label="Open shopping cart"
>
<span class="floating-cart-icon" aria-hidden="true">&#128722;</span>
{badge_html}
</button>
</div>
<div class="floating-cart-panel" id="floating-cart-panel" popover>
<p class="floating-cart-title">Shopping cart</p>
<p class="floating-cart-summary">{total_items} items · {remaining_items} remaining</p>
{item_markup}
<button
    class="floating-cart-link"
    type="button"
    data-floating-cart-open-list="true"
>
Open full shopping list
</button>
</div>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        f"""
        <script>
        (() => {{
            const targetLabel = {SHOPPING_LIST_SWITCH_LABEL!r};
            const attach = () => {{
                const doc = window.parent.document;
                const openButton = doc.querySelector('[data-floating-cart-open-list="true"]');
                if (!openButton) {{
                    window.setTimeout(attach, 80);
                    return;
                }}

                openButton.onclick = () => {{
                    const buttons = Array.from(doc.querySelectorAll('button'));
                    const switchButton = buttons.find((button) =>
                        button.textContent.trim() === targetLabel
                    );
                    if (switchButton) {{
                        switchButton.click();
                    }}
                }};
            }};
            attach();
        }})();
        </script>
        """,
        height=0,
    )
    st.markdown(
        """
        <style>
        div[data-testid="stElementContainer"]:has(.floating-cart-switch-anchor) {
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: hidden !important;
        }

        div[data-testid="stElementContainer"]:has(.floating-cart-switch-anchor)
        + div[data-testid="stElementContainer"] {
            position: fixed !important;
            top: -100px !important;
            left: -100px !important;
            width: 1px !important;
            height: 1px !important;
            opacity: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }
        </style>
        <span class="floating-cart-switch-anchor"></span>
        """,
        unsafe_allow_html=True,
    )
    if st.button(SHOPPING_LIST_SWITCH_LABEL, key="floating_cart_switch_page"):
        st.switch_page(SHOPPING_LIST_PAGE)
