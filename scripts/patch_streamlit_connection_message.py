#!/usr/bin/env python3
"""Patch Streamlit's local connection-error hint for this project."""

from __future__ import annotations

from pathlib import Path

OLD_COMMAND = "streamlit run yourscript.py"
NEW_COMMAND = "streamlit run app/streamlit_app.py"


def find_streamlit_static_dir() -> Path:
    import streamlit

    return Path(streamlit.__file__).resolve().parent / "static" / "static" / "js"


def patch_connection_message(static_dir: Path | None = None) -> Path:
    """Replace Streamlit's generic restart command in its compiled frontend bundle."""
    static_dir = static_dir or find_streamlit_static_dir()
    candidates = sorted(static_dir.glob("index.*.js"))

    for candidate in candidates:
        text = candidate.read_text()
        if NEW_COMMAND in text:
            return candidate
        if OLD_COMMAND not in text:
            continue

        candidate.write_text(text.replace(OLD_COMMAND, NEW_COMMAND))
        return candidate

    raise FileNotFoundError(
        "Could not find Streamlit's frontend bundle containing "
        f"{OLD_COMMAND!r}. Streamlit may have changed this message."
    )


def main() -> int:
    patched = patch_connection_message()
    print(f"Streamlit connection hint patched in {patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
