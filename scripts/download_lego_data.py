#!/usr/bin/env python3
"""
scripts/download_lego_data.py
Downloads the classic synthetic NeRF Lego tractor dataset (Mildenhall et al., ECCV 2020)
from high-speed CDN mirrors using standard Python library (no pip packages needed).

Usage:
    python3 scripts/download_lego_data.py
    python3 scripts/download_lego_data.py --out-dir data/lego --splits train val test --workers 16
"""

import os
import sys
import json
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HF_REPO_API = "https://huggingface.co/api/datasets/rishitdagli/nerf-gs-datasets/tree/main/lego"
HF_RAW_BASE = "https://huggingface.co/datasets/rishitdagli/nerf-gs-datasets/resolve/main/lego"

HEADERS = {
    "User-Agent": "WLNeRFLite-LegoGrabber/1.0"
}


def download_single_file(rel_path: str, out_dir: str, overwrite: bool = False):
    dest_path = os.path.join(out_dir, rel_path)
    if not overwrite and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return rel_path, "cached", os.path.getsize(dest_path)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    url = f"{HF_RAW_BASE}/{rel_path}"

    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        content = resp.read()
        f.write(content)
        size = len(content)

    return rel_path, "downloaded", size


def fetch_file_list(subfolder: str) -> list:
    url = f"{HF_REPO_API}/{subfolder}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    files = []
    for item in data:
        if item.get("type") == "file":
            p = item["path"]
            if p.startswith("lego/"):
                p = p[len("lego/"):]
            files.append(p)
    return files


def main():
    parser = argparse.ArgumentParser(description="Download the classic NeRF Lego tractor dataset.")
    parser.add_argument("--out-dir", default="data/lego", help="Destination folder (default: data/lego)")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test", "all"],
        help="Dataset splits to download (default: train val)",
    )
    parser.add_argument("--workers", type=int, default=16, help="Concurrent worker threads (default: 16)")
    parser.add_argument("--overwrite", action="store_true", help="Force re-download existing files")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    splits = ["train", "val", "test"] if "all" in args.splits else args.splits

    print("==================================================================")
    print("WLNeRFLite Lego Tractor Dataset Downloader")
    print("==================================================================")
    print(f"Target directory : {out_dir}")
    print(f"Selected splits  : {', '.join(splits)}")
    print(f"Download workers : {args.workers}")
    print("------------------------------------------------------------------")

    # 1. Download camera transforms JSON files
    json_targets = []
    if "train" in splits:
        json_targets.append("transforms_train.json")
    if "val" in splits:
        json_targets.append("transforms_val.json")
    if "test" in splits:
        json_targets.append("transforms_test.json")

    print("[1/2] Fetching camera transforms JSON files...")
    for jf in json_targets:
        rel, status, size = download_single_file(jf, out_dir, overwrite=args.overwrite)
        print(f"  -> {jf} ({size / 1024:.1f} KB, {status})")

    # 2. Gather list of image files for selected splits
    print("\n[2/2] Fetching image file manifests and downloading frames...")
    all_image_files = []
    for s in splits:
        try:
            split_files = fetch_file_list(s)
            print(f"  Discovered {len(split_files)} frames in '{s}' split.")
            all_image_files.extend(split_files)
        except Exception as e:
            print(f"  Warning: Failed to fetch directory manifest for split '{s}': {e}")

    total_files = len(all_image_files)
    if total_files == 0:
        print("No image files found to download. Exiting.")
        sys.exit(1)

    print(f"\nStarting concurrent download of {total_files} images into {out_dir}...")
    completed = 0
    total_bytes = 0
    cached_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_single_file, rel_file, out_dir, args.overwrite): rel_file
            for rel_file in all_image_files
        }

        for fut in as_completed(futures):
            rel_file = futures[fut]
            try:
                rel, status, size = fut.result()
                completed += 1
                total_bytes += size
                if status == "cached":
                    cached_count += 1
                if completed % 25 == 0 or completed == total_files:
                    pct = (completed / total_files) * 100
                    print(
                        f"  [{completed}/{total_files}] ({pct:5.1f}%) | "
                        f"{(total_bytes / (1024 * 1024)):5.1f} MB processed"
                    )
            except Exception as e:
                print(f"  Error downloading {rel_file}: {e}")

    print("\n------------------------------------------------------------------")
    print(f"Dataset download complete!")
    print(f"  Output location : {out_dir}")
    print(f"  Total files     : {completed} ({cached_count} cached)")
    print(f"  Total size      : {total_bytes / (1024 * 1024):.1f} MB")
    print("==================================================================")


if __name__ == "__main__":
    main()

