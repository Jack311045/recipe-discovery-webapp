"""Shared visual theme helpers for Streamlit pages."""

from __future__ import annotations

import streamlit as st


def apply_restaurant_menu_theme() -> None:
    """Inject a warm, menu-inspired visual theme into the current page."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Allura&family=Cormorant+Garamond:wght@400;500;600;700&family=Lora:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

        :root {
            --menu-bg: #F6EFD9;
            --menu-bg-soft: #FFF9EF;
            --menu-border: #C8B38A;
            --menu-text: #3B2F2A;
            --menu-accent: #6B4F3A;
            --menu-accent-soft: #556B56;
            --menu-shadow: rgba(70, 48, 34, 0.09);
        }

        .stApp {
            background:
                radial-gradient(circle at 4% 0%, #fffdfa 0%, rgba(255, 249, 239, 0.82) 28%, rgba(246, 239, 217, 0.94) 62%, #efe3c7 100%);
            color: var(--menu-text);
            font-family: "Lora", "Merriweather", serif;
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 1.35rem;
            padding-bottom: 2.8rem;
        }

        .stApp h1,
        .stApp h2,
        .stApp h3,
        .stApp h4,
        .stApp [data-testid="stMarkdownContainer"] h1,
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3 {
            color: var(--menu-accent);
            letter-spacing: 0.015em;
            line-height: 1.2;
        }

        .stApp h1,
        .stApp [data-testid="stMarkdownContainer"] h1 {
            font-family: "Playfair Display", "Cormorant Garamond", serif;
            font-weight: 600;
        }

        .stApp h2,
        .stApp h3,
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3 {
            font-family: "Cormorant Garamond", "Playfair Display", serif;
            font-weight: 600;
        }

        .stApp p,
        .stApp li,
        .stApp label,
        .stApp input,
        .stApp textarea {
            font-family: "Lora", "Merriweather", serif;
            color: var(--menu-text);
            font-size: 1.02rem;
            line-height: 1.52;
        }

        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(255, 250, 241, 0.96) 0%, rgba(244, 233, 205, 0.96) 100%);
            border-right: 1px solid var(--menu-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
            color: #5f4937;
            font-size: 1rem;
        }

        .stTextInput > div > div > input,
        .stTextArea textarea,
        .stNumberInput input,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-testid="stFileUploader"] section {
            background: #fffdf7 !important;
            border: 1px solid var(--menu-border) !important;
            border-radius: 0.7rem !important;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
        }

        .stTextInput > div > div > input:focus,
        .stTextArea textarea:focus,
        [data-baseweb="input"] > div:focus-within,
        [data-baseweb="select"] > div:focus-within {
            border-color: #8d6f50 !important;
            box-shadow: 0 0 0 0.12rem rgba(107, 79, 58, 0.18) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        .stLinkButton > a {
            background: linear-gradient(180deg, #7a5a43 0%, #6B4F3A 100%);
            color: #ffffff !important;
            border: 1px solid #5A422F !important;
            border-radius: 999px !important;
            font-weight: 600;
            font-size: 1rem;
            min-height: 2.55rem;
            letter-spacing: 0.015em;
            text-shadow: 0 1px 0 rgba(32, 20, 13, 0.45);
            box-shadow: 0 2px 8px rgba(68, 45, 29, 0.22);
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
        }

        .stButton > button *,
        .stDownloadButton > button *,
        .stLinkButton > a * {
            color: #ffffff !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        .stLinkButton > a:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 14px rgba(68, 45, 29, 0.26);
            filter: brightness(1.03);
        }

        .stButton > button:active,
        .stDownloadButton > button:active,
        .stLinkButton > a:active {
            transform: translateY(0);
        }

        .stButton > button:focus,
        .stDownloadButton > button:focus,
        .stLinkButton > a:focus {
            box-shadow: 0 0 0 0.18rem rgba(107, 79, 58, 0.28) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(180deg, rgba(255, 249, 239, 0.97) 0%, rgba(255, 245, 228, 0.97) 100%);
            border: 1px solid var(--menu-border);
            border-radius: 0.95rem;
            box-shadow: 0 8px 22px var(--menu-shadow);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 0.35rem 0.4rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255, 250, 241, 0.94);
            border: 1px solid var(--menu-border);
            border-radius: 0.75rem;
            padding: 0.72rem 0.72rem;
            box-shadow: 0 2px 8px rgba(88, 60, 40, 0.07);
        }

        div[data-testid="stMetricLabel"] p {
            font-size: 0.98rem !important;
            color: #5b4434 !important;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.45rem !important;
            color: #3b2f2a !important;
            line-height: 1.1 !important;
        }

        div[data-testid="stTabs"] {
            margin-top: 0.35rem;
        }

        div[data-testid="stTabs"] [role="tablist"] {
            display: grid !important;
            grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
            gap: 0.2rem;
            width: 100%;
        }

        button[data-baseweb="tab"] {
            width: 100% !important;
            justify-content: center !important;
            border-radius: 0.55rem 0.55rem 0 0 !important;
            border: 1px solid transparent !important;
            color: #5e4838 !important;
            font-weight: 600;
            font-size: 1rem !important;
            min-height: 2.2rem;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            border-color: var(--menu-border) !important;
            background: rgba(255, 249, 239, 0.95) !important;
            color: var(--menu-accent) !important;
        }

        .menu-card-reviews {
            display: grid;
            gap: 0.55rem;
            max-height: 18rem;
            overflow-y: auto;
            padding-right: 0.2rem;
        }

        .menu-card-review {
            padding: 0.62rem 0.7rem;
            border: 1px solid rgba(200, 179, 138, 0.82);
            border-radius: 0.65rem;
            background: rgba(255, 253, 247, 0.88);
            color: #3b2f2a;
            font-size: 0.95rem;
            line-height: 1.45;
        }

        .menu-card-empty-reviews {
            min-height: 7rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px dashed rgba(183, 159, 115, 0.78);
            border-radius: 0.65rem;
            background: rgba(255, 253, 247, 0.64);
            color: #746152;
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
        }

        div[data-testid="stExpander"] > details {
            background: #fffaf1;
            border: 1px solid var(--menu-border);
            border-radius: 0.75rem;
        }

        div[data-testid="stExpander"] > details > summary {
            color: #5e4838;
            font-weight: 600;
        }

        .stAlert {
            border-radius: 0.7rem;
            border: 1px solid var(--menu-border);
        }

        hr,
        [data-testid="stMarkdownContainer"] hr {
            border: none;
            border-top: 1px solid rgba(200, 179, 138, 0.75);
            margin-top: 0.9rem;
            margin-bottom: 0.9rem;
        }

        .menu-script-accent {
            font-family: "Allura", cursive;
            color: #7b6049;
            font-size: 1.45rem;
            line-height: 1;
            margin-bottom: 0.05rem;
            letter-spacing: 0.02em;
        }

        .menu-page-header {
            padding: 1rem 1.2rem;
            margin-bottom: 0.7rem;
            border: 1px solid rgba(200, 179, 138, 0.9);
            border-radius: 0.9rem;
            background: linear-gradient(120deg, rgba(255, 249, 239, 0.98) 0%, rgba(245, 233, 203, 0.93) 100%);
            box-shadow: 0 7px 20px rgba(84, 56, 36, 0.1);
        }

        .menu-page-kicker {
            margin: 0;
            font-family: "Allura", cursive;
            font-size: 1.45rem;
            color: #7b6049;
            line-height: 1;
        }

        .menu-page-title {
            margin: 0.22rem 0 0.28rem 0;
            font-family: "Playfair Display", "Cormorant Garamond", serif;
            color: #5b4434;
            font-size: 2rem;
            line-height: 1.15;
            letter-spacing: 0.01em;
        }

        .menu-page-subtitle {
            margin: 0;
            color: #4f3f34;
            font-size: 1.05rem;
            line-height: 1.5;
            max-width: 72ch;
        }

        .search-guidance-text {
            margin: 0.08rem 0 0.65rem 0;
            color: #5a483d;
            font-size: 0.98rem;
        }

        .search-hero {
            padding: 1.05rem 1.2rem;
            border-radius: 0.95rem;
            border: 1px solid var(--menu-border);
            background:
                linear-gradient(120deg, rgba(255, 249, 239, 0.98) 0%, rgba(245, 233, 203, 0.93) 100%);
            box-shadow: 0 7px 20px rgba(84, 56, 36, 0.1);
            margin-bottom: 0.5rem;
            position: relative;
            overflow: hidden;
        }

        .search-hero::after {
            content: "";
            position: absolute;
            left: 1rem;
            right: 1rem;
            bottom: 0.6rem;
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(107, 79, 58, 0.5) 45%, transparent 100%);
        }

        .search-hero .search-kicker {
            font-family: "Allura", cursive;
            color: #7b6049;
            font-size: 1.35rem;
            margin: 0;
            line-height: 1;
        }

        .search-hero h2 {
            margin: 0.2rem 0 0.18rem 0;
            color: #5b4434;
            font-size: 1.75rem;
        }

        .search-hero p {
            margin: 0.2rem 0 0 0;
            color: #4a3a32;
            max-width: 62ch;
            font-size: 1.03rem;
        }

        .menu-card-title {
            margin: 0;
            font-family: "Playfair Display", "Cormorant Garamond", serif;
            color: #5b4434;
            font-size: 1.42rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            line-height: 1.25;
        }

        .menu-card-meta {
            margin: 0.18rem 0 0.35rem 0;
            color: #665244;
            font-size: 1rem;
        }

        .menu-card-description {
            margin: 0.2rem 0 0.4rem 0;
            color: #3f322c;
            font-size: 1.02rem;
            line-height: 1.55;
        }

        .menu-card-overview-meta {
            margin: 0.2rem 0 0.45rem 0;
            color: #5e4a3e;
            font-size: 0.96rem;
            line-height: 1.42;
            font-weight: 600;
        }

        .menu-card-image-placeholder {
            min-height: 7.5rem;
            border: 1px dashed rgba(183, 159, 115, 0.78);
            border-radius: 0.65rem;
            background: rgba(255, 253, 247, 0.68);
            color: #7b6a5b;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.95rem;
            font-weight: 600;
        }

        .menu-card-signal-row {
            margin: 0.35rem 0 0 0;
            color: #665244;
            font-size: 1rem;
            line-height: 1.35;
        }

        .menu-card-meta-text {
            display: inline-flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.2rem 0.38rem;
        }

        .menu-match-inline {
            display: inline-flex;
            align-items: center;
            min-height: 1.25rem;
            padding: 0.06rem 0.42rem;
            border: 1px solid rgba(183, 159, 115, 0.92);
            border-radius: 999px;
            background: rgba(255, 250, 241, 0.94);
            color: #5b4434;
            font-size: 0.9rem;
            font-weight: 700;
            white-space: nowrap;
            line-height: 1.1;
        }

        .menu-match-badge {
            display: inline-flex;
            flex-direction: column;
            align-items: flex-end;
            justify-content: center;
            min-width: 4.5rem;
            padding: 0.22rem 0.42rem;
            border: 1px solid var(--menu-border);
            border-radius: 0.55rem;
            background: rgba(255, 250, 241, 0.92);
            white-space: nowrap;
        }

        .menu-match-badge span {
            font-size: 0.62rem;
            line-height: 1;
            color: #6c5848;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .menu-match-badge strong {
            display: block;
            margin-top: 0.08rem;
            font-size: 0.82rem;
            line-height: 1.1;
            color: #5b4434;
            font-weight: 700;
        }

        .menu-card-divider {
            height: 1px;
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
            background: linear-gradient(90deg, rgba(200, 179, 138, 0.15), rgba(200, 179, 138, 0.9), rgba(200, 179, 138, 0.15));
        }

        [data-testid="stCaptionContainer"] {
            color: #6d584b;
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: #6c5848;
        }

        @media (max-width: 900px) {
            .main .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .search-hero h2 {
                font-size: 1.45rem;
            }

            .menu-card-title {
                font-size: 1.22rem;
            }

            .menu-page-title {
                font-size: 1.62rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
