#!/usr/bin/env python3
"""fetch_uci_data.py

Download the UCI *Individual Household Electric Power Consumption* dataset
and extract it into the project's data directory.

Usage
-----
$ python scripts/fetch_uci_data.py  # default location data/raw/uci
$ python scripts/fetch_uci_data.py --output_dir path/to/dir
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile

DATA_URL: str = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)


def _download(url: str, dest: pathlib.Path) -> None:
    """Stream-download *url* into *dest* (shows basic progress)."""
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the UCI household power consumption dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="data/raw/uci",
        help="Directory where the dataset will be unpacked (created if needed)",
    )
    args = parser.parse_args(argv)

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_zip = pathlib.Path(tmpdir) / "uci_power.zip"
        print(f"Downloading dataset to {tmp_zip} …")
        _download(DATA_URL, tmp_zip)
        print("Download complete. Extracting …")

        with zipfile.ZipFile(tmp_zip) as zf:
            zf.extractall(output_dir)
        print(f"Dataset extracted to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
