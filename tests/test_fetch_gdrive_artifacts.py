from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_gdrive_artifacts.py"
SPEC = importlib.util.spec_from_file_location("fetch_gdrive_artifacts", SCRIPT_PATH)
assert SPEC is not None
fetch_gdrive_artifacts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fetch_gdrive_artifacts)


def test_extract_folder_id_accepts_google_folder_url_and_id() -> None:
    folder_id = "1bzKGQINcPabu0nIFnEqkJQwMlJgVRPLK"

    assert (
        fetch_gdrive_artifacts.extract_folder_id(
            f"https://drive.google.com/drive/folders/{folder_id}?usp=sharing"
        )
        == folder_id
    )
    open_url = f"https://drive.google.com/open?id={folder_id}"
    assert fetch_gdrive_artifacts.extract_folder_id(open_url) == folder_id
    assert fetch_gdrive_artifacts.extract_folder_id(folder_id) == folder_id
    assert fetch_gdrive_artifacts.extract_folder_id("not a folder") is None


def test_install_downloaded_files_places_processed_csv_and_artifacts(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "Processed_data_updated2.csv").write_text("recipe_id,name\n1,Soup\n")
    (staging / "recipe_embeddings.npy").write_bytes(b"embedding bytes")
    (staging / "recipe_ids.csv").write_text("recipe_id\n1\n")

    processed_path = tmp_path / "data" / "processed" / "Processed_data_updated2.csv"
    artifact_dir = tmp_path / "data" / "artifacts"

    installed_processed, installed_artifacts = fetch_gdrive_artifacts.install_downloaded_files(
        staging_dir=staging,
        artifact_dir=artifact_dir,
        processed_path=processed_path,
    )

    assert installed_processed == processed_path
    assert processed_path.read_text() == "recipe_id,name\n1,Soup\n"
    assert (artifact_dir / "recipe_embeddings.npy").read_bytes() == b"embedding bytes"
    assert (artifact_dir / "recipe_ids.csv").read_text() == "recipe_id\n1\n"
    assert sorted(path.name for path in installed_artifacts) == [
        "recipe_embeddings.npy",
        "recipe_ids.csv",
    ]


def test_install_downloaded_files_flattens_nested_downloaded_artifacts(tmp_path: Path) -> None:
    nested = tmp_path / "staging" / "shared-folder" / "nested"
    nested.mkdir(parents=True)
    (nested / "Processed_data_updated2.csv").write_text("recipe_id,name\n2,Salad\n")
    (nested / "embedding_metadata.json").write_text('{"rows": 1}\n')

    processed_path = tmp_path / "processed" / "Processed_data_updated2.csv"
    artifact_dir = tmp_path / "artifacts"

    fetch_gdrive_artifacts.install_downloaded_files(
        staging_dir=tmp_path / "staging",
        artifact_dir=artifact_dir,
        processed_path=processed_path,
    )

    assert processed_path.read_text() == "recipe_id,name\n2,Salad\n"
    assert (artifact_dir / "embedding_metadata.json").read_text() == '{"rows": 1}\n'


def test_install_downloaded_files_extracts_processed_csv_zip(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    zip_path = staging / "Processed_data_updated2.csv.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("Processed_data_updated2.csv", "recipe_id,name\n3,Pasta\n")
    (staging / "recipe_ids.csv").write_text("recipe_id\n3\n")

    processed_path = tmp_path / "processed" / "Processed_data_updated2.csv"
    artifact_dir = tmp_path / "artifacts"

    _, installed_artifacts = fetch_gdrive_artifacts.install_downloaded_files(
        staging_dir=staging,
        artifact_dir=artifact_dir,
        processed_path=processed_path,
    )

    assert processed_path.read_text() == "recipe_id,name\n3,Pasta\n"
    assert (artifact_dir / "recipe_ids.csv").read_text() == "recipe_id\n3\n"
    assert sorted(path.name for path in installed_artifacts) == ["recipe_ids.csv"]
    assert not (artifact_dir / "Processed_data_updated2.csv.zip").exists()


def test_install_downloaded_files_overwrites_existing_files(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "Processed_data_updated2.csv").write_text("new processed\n")
    (staging / "recipe_ids.csv").write_text("new artifact\n")

    processed_path = tmp_path / "data" / "processed" / "Processed_data_updated2.csv"
    artifact_dir = tmp_path / "data" / "artifacts"
    processed_path.parent.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    processed_path.write_text("old processed\n")
    (artifact_dir / "recipe_ids.csv").write_text("old artifact\n")

    fetch_gdrive_artifacts.install_downloaded_files(
        staging_dir=staging,
        artifact_dir=artifact_dir,
        processed_path=processed_path,
    )

    assert processed_path.read_text() == "new processed\n"
    assert (artifact_dir / "recipe_ids.csv").read_text() == "new artifact\n"


def test_install_downloaded_files_raises_when_processed_csv_is_missing(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "recipe_ids.csv").write_text("recipe_id\n1\n")

    with pytest.raises(FileNotFoundError, match="Processed_data_updated2.csv"):
        fetch_gdrive_artifacts.install_downloaded_files(
            staging_dir=staging,
            artifact_dir=tmp_path / "artifacts",
            processed_path=tmp_path / "processed" / "Processed_data_updated2.csv",
        )
