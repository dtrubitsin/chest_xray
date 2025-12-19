#!/usr/bin/env python3
"""
Download ChestXray NIHCC image archives and bounding-box annotations.

Optionally extracts the downloaded archives into an images directory. By
default files are stored under `data/raw`, which keeps large artifacts out of
the repository root.
"""

from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

IMAGE_LINKS = [
    "https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz",
    "https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz",
    "https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz",
    "https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz",
    "https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz",
    "https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz",
    "https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz",
    "https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz",
    "https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz",
    "https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz",
    "https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz",
    "https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz",
]


def download_file(url: str, destination: Path, overwrite: bool = False) -> None:
    """
    Download a single file if it does not already exist.

    :param url: Remote URL to download.
    :param destination: Local path to store the downloaded file.
    :param overwrite: Whether to re-download when the destination exists.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[skip] {destination} exists; use --overwrite to re-download.")
        return
    print(f"[download] {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def download_archives(raw_dir: Path, start: int, end: int | None, overwrite: bool) -> None:
    """
    Download image archives in the requested range.

    :param raw_dir: Directory to hold downloaded archives.
    :param start: First archive index (1-based).
    :param end: Last archive index (inclusive). If None, downloads to the end.
    :param overwrite: Whether to re-download existing files.
    """
    end = end or len(IMAGE_LINKS)
    subset = IMAGE_LINKS[start - 1 : end]
    for idx, link in enumerate(subset, start=start):
        destination = raw_dir / f"images_{idx:02d}.tar.gz"
        download_file(link, destination, overwrite=overwrite)


def _select_members(members: Iterable[tarfile.TarInfo]) -> list[tarfile.TarInfo]:
    """
    Filter out directories to avoid creating empty entries during extraction.

    :param members: Tar members to filter.
    :return: File-only members.
    """
    return [m for m in members if m.isfile()]


def extract_archives(raw_dir: Path, extract_dir: Path) -> None:
    """
    Extract all image archives found under `raw_dir` into `extract_dir`.

    :param raw_dir: Directory containing downloaded archives.
    :param extract_dir: Destination directory for extracted images.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)
    archives = sorted(raw_dir.glob("images_*.tar.gz"))
    if not archives:
        print("[extract] no archives found under", raw_dir)
        return

    for archive in archives:
        print(f"[extract] {archive} -> {extract_dir}")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=extract_dir, members=_select_members(tar))


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Download NIH ChestXray archives and bbox file.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Directory for downloaded archives/CSV.")
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("data"),
        help="Directory to extract image archives into.",
    )
    parser.add_argument("--start", type=int, default=1, help="First archive index to download (1-based).")
    parser.add_argument("--end", type=int, default=None, help="Last archive index to download (inclusive).")
    parser.add_argument("--overwrite", action="store_true", help="Re-download files even if they exist.")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded archives to --extract-dir.")
    return parser.parse_args()


def main() -> None:
    """
    Execute downloads and optional extraction.
    """
    args = parse_args()
    raw_dir: Path = args.raw_dir
    extract_dir: Path = args.extract_dir

    download_archives(raw_dir, start=args.start, end=args.end, overwrite=args.overwrite)

    if args.extract:
        extract_archives(raw_dir, extract_dir)
    print("Done. Verify checksums if provided by the data source.")


if __name__ == "__main__":
    main()