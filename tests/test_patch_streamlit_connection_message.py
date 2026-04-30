from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "patch_streamlit_connection_message.py"
)
SPEC = importlib.util.spec_from_file_location("patch_streamlit_connection_message", SCRIPT_PATH)
assert SPEC is not None
patch_streamlit_connection_message = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(patch_streamlit_connection_message)


def test_patch_connection_message_replaces_generic_command(tmp_path: Path) -> None:
    bundle = tmp_path / "index.test.js"
    bundle.write_text("codeBlock:`streamlit run yourscript.py`")

    patched = patch_streamlit_connection_message.patch_connection_message(tmp_path)

    assert patched == bundle
    assert bundle.read_text() == "codeBlock:`streamlit run app/streamlit_app.py`"


def test_patch_connection_message_is_idempotent(tmp_path: Path) -> None:
    bundle = tmp_path / "index.test.js"
    bundle.write_text("codeBlock:`streamlit run app/streamlit_app.py`")

    patched = patch_streamlit_connection_message.patch_connection_message(tmp_path)

    assert patched == bundle
    assert bundle.read_text() == "codeBlock:`streamlit run app/streamlit_app.py`"
