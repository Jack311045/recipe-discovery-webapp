#!/usr/bin/env python3
"""Download processed data and embedding artifacts from a shared Google Drive folder."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROCESSED_FILENAME = "Processed_data_updated2.csv"
PROCESSED_ZIP_FILENAME = f"{PROCESSED_FILENAME}.zip"
DEFAULT_PROCESSED_PATH = Path("data/processed") / PROCESSED_FILENAME


def extract_folder_id(value: str) -> str | None:
    if not value:
        return None
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]+)", value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", value):
        return value
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch processed data and embedding artifacts from a shared Google Drive folder."
        )
    )
    parser.add_argument(
        "--folder",
        default=os.getenv("GDRIVE_ARTIFACTS_FOLDER"),
        help="Google Drive folder link or folder ID."
        " Can also be set via GDRIVE_ARTIFACTS_FOLDER.",
    )
    parser.add_argument(
        "--out",
        default="data/artifacts",
        help="Destination folder for artifacts.",
    )
    parser.add_argument(
        "--allow-cookies",
        action="store_true",
        help="Allow gdown to use cookies if needed for permissions.",
    )
    return parser.parse_args()


def install_downloaded_files(
    staging_dir: Path,
    artifact_dir: Path,
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    processed_filename: str = PROCESSED_FILENAME,
) -> tuple[Path, list[Path]]:
    """Place the processed CSV and downloaded artifacts in their app locations."""
    staging_dir = Path(staging_dir)
    artifact_dir = Path(artifact_dir)
    processed_path = Path(processed_path)

    processed_source: Path | None = None
    processed_zip_source: Path | None = None
    artifact_sources: list[Path] = []
    for path in sorted(staging_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == processed_filename:
            processed_source = path
        elif path.name == f"{processed_filename}.zip":
            processed_zip_source = path
        else:
            artifact_sources.append(path)

    if processed_source is None and processed_zip_source is None:
        raise FileNotFoundError(
            "Downloaded folder did not contain required processed CSV: "
            f"{processed_filename} or {processed_filename}.zip"
        )

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if processed_source is not None:
        shutil.copy2(processed_source, processed_path)
    else:
        assert processed_zip_source is not None
        with zipfile.ZipFile(processed_zip_source) as archive:
            try:
                member = next(
                    name for name in archive.namelist() if Path(name).name == processed_filename
                )
            except StopIteration as exc:
                raise FileNotFoundError(
                    f"Processed ZIP did not contain required CSV: {processed_filename}"
                ) from exc
            with archive.open(member) as source, processed_path.open("wb") as destination:
                shutil.copyfileobj(source, destination)

    installed_artifacts: list[Path] = []
    for source in artifact_sources:
        destination = artifact_dir / source.name
        shutil.copy2(source, destination)
        installed_artifacts.append(destination)

    return processed_path, installed_artifacts


def main() -> int:
    args = parse_args()
    folder_id = extract_folder_id(args.folder or "")
    if not folder_id:
        print(
            "Missing or invalid Google Drive folder. Provide --folder or set "
            "GDRIVE_ARTIFACTS_FOLDER.",
            file=sys.stderr,
        )
        return 2

    try:
        import gdown
    except ImportError:
        print(
            "Missing dependency: gdown. Install with `pip install gdown`.",
            file=sys.stderr,
        )
        return 3

    artifact_dir = Path(args.out)
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(
        f"Downloading processed data and artifacts from {folder_url} "
        "into a temporary staging folder"
    )

    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = Path(tmp)
        gdown.download_folder(
            folder_url,
            output=str(staging_dir),
            quiet=False,
            use_cookies=args.allow_cookies,
        )
        try:
            processed_path, artifacts = install_downloaded_files(
                staging_dir=staging_dir,
                artifact_dir=artifact_dir,
            )
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 4

    print(f"Installed processed CSV -> {processed_path}")
    print(f"Installed {len(artifacts)} artifact file(s) -> {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
