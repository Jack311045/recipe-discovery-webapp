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
        .stApp span,
        .stApp div,
        .stApp input,
        .stApp textarea,
        .stApp button {
            font-family: "Lora", "Merriweather", serif;
            color: var(--menu-text);
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
            color: #FFF9EF !important;
            border: 1px solid #5A422F !important;
            border-radius: 999px !important;
            font-weight: 600;
            letter-spacing: 0.015em;
            box-shadow: 0 2px 8px rgba(68, 45, 29, 0.22);
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
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
            padding: 0.55rem 0.6rem;
            box-shadow: 0 2px 8px rgba(88, 60, 40, 0.07);
        }

        div[data-testid="stTabs"] {
            margin-top: 0.35rem;
        }

        button[data-baseweb="tab"] {
            border-radius: 0.55rem 0.55rem 0 0 !important;
            border: 1px solid transparent !important;
            color: #5e4838 !important;
            font-weight: 600;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            border-color: var(--menu-border) !important;
            background: rgba(255, 249, 239, 0.95) !important;
            color: var(--menu-accent) !important;
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
            font-size: 1.25rem;
            line-height: 1;
            margin-bottom: 0.05rem;
            letter-spacing: 0.02em;
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
        }

        .search-hero p {
            margin: 0.2rem 0 0 0;
            color: #4a3a32;
            max-width: 62ch;
        }

        .menu-card-title {
            margin: 0;
            font-family: "Playfair Display", "Cormorant Garamond", serif;
            color: #5b4434;
            font-size: 1.32rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .menu-card-meta {
            margin: 0.18rem 0 0.35rem 0;
            color: #665244;
            font-size: 0.94rem;
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
                font-size: 1.16rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
