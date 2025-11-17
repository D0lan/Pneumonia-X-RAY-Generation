#!/usr/bin/env python3
"""
Download and unzip the Kaggle Chest X-Ray Pneumonia dataset:

    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Usage:
    python download_chestxray_kaggle.py --out_dir /home/dolan/data

Result:
    /home/dolan/data/chest_xray/
        train/
          NORMAL/
          PNEUMONIA/
        val/
        test/
"""

import argparse
from pathlib import Path
import zipfile

from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_SLUG = "paultimothymooney/chest-xray-pneumonia"


def download_chestxray(out_dir: str) -> Path:
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"[+] Downloading {DATASET_SLUG} to {out_path} ...")
    api.dataset_download_files(
        DATASET_SLUG,
        path=str(out_path),
        quiet=False,
        unzip=False,   # we’ll unzip manually
    )

    # Kaggle usually names it something like 'chest-xray-pneumonia.zip' or 'dataset.zip'
    zips = list(out_path.glob("*.zip"))
    if not zips:
        raise RuntimeError(f"No .zip files found in {out_path} after download.")
    zip_path = max(zips, key=lambda p: p.stat().st_mtime)
    print(f"[+] Unzipping {zip_path} ...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_path)

    # Find the chest_xray root (in case Kaggle changed top-level folder name)
    chest_root = None
    for p in out_path.rglob("chest_xray"):
        if (p / "train").exists():
            chest_root = p
            break
    if chest_root is None:
        # fallback: maybe the zip directly extracted train/val/test in out_path
        if (out_path / "train").exists():
            chest_root = out_path
        else:
            raise RuntimeError("Could not locate 'chest_xray/train' after unzip.")

    print(f"[+] Dataset ready at: {chest_root}")
    print("    Subdirs:", [d.name for d in chest_root.iterdir() if d.is_dir()])
    return chest_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory where the dataset zip will be downloaded and extracted.",
    )
    args = parser.parse_args()

    download_chestxray(args.out_dir)


if __name__ == "__main__":
    main()
