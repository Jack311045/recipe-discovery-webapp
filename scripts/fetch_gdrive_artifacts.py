#!/usr/bin/env python3
"""Download embedding artifacts from a shared Google Drive folder."""

from __future__ import annotations

import argparse
import os
import re
import sys


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
        description="Fetch embedding artifacts from a shared Google Drive folder."
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

    os.makedirs(args.out, exist_ok=True)
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading artifacts from {folder_url} -> {args.out}")

    gdown.download_folder(
        folder_url,
        output=args.out,
        quiet=False,
        use_cookies=args.allow_cookies,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
