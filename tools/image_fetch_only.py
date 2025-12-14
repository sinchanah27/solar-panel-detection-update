#!/usr/bin/env python3
"""
tools/image_fetch_only.py

Fetch satellite images ONLY from coordinates file (Excel/CSV).
Auto-detects whether rows are (lat, lon) or (lon, lat).
Usage:
  python -m tools.image_fetch_only --input path/to/file.xlsx --outdir dataset_raw --config config/config.yaml --limit 5
"""

import argparse
import pandas as pd
import os
import json
from pathlib import Path
import logging
from typing import Tuple, Optional

from agents.satellite_fetch_agent import SatelliteFetchAgent
from controllers.master_controller import load_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

def detect_lat_lon_pair(a: float, b: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Given two numeric values, decide which is lat and which is lon.
    Returns (lat, lon) or (None, None) if invalid.
    """
    try:
        a_f = float(a)
        b_f = float(b)
    except Exception:
        return None, None

    # Case: (lat, lon)
    if -90 <= a_f <= 90 and -180 <= b_f <= 180:
        return a_f, b_f
    # Case: (lon, lat)
    if -180 <= a_f <= 180 and -90 <= b_f <= 90:
        return b_f, a_f
    return None, None

def find_coord_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (col_a, col_b) representing the two columns to read for coordinates.
    Preference:
      - explicit 'latitude' and 'longitude' case-insensitive
      - if not present, find any two numeric columns (first pair)
    """
    cols_lower = [c.strip().lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}

    if "latitude" in cols_lower and "longitude" in cols_lower:
        return col_map["latitude"], col_map["longitude"]

    # Common alternate names
    if "lat" in cols_lower and "lon" in cols_lower:
        return col_map["lat"], col_map["lon"]
    # try pairs of numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    raise ValueError("Could not find coordinate columns. Provide 'latitude' and 'longitude' or two numeric columns.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Excel/CSV with coordinates")
    parser.add_argument("--sheet", default=None, help="Excel sheet name (optional)")
    parser.add_argument("--outdir", "-o", default="dataset_raw", help="Folder to save images")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--limit", type=int, default=None, help="Only fetch first N rows")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    imgdir = outdir / "images"
    imgdir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)

    fpath = Path(args.input)
    if not fpath.exists():
        raise FileNotFoundError(f"Input file not found: {fpath}")

    # load sheet-aware Excel or CSV
    if fpath.suffix.lower() in [".xlsx", ".xls"]:
        # read either specified sheet or all sheets
        x = pd.read_excel(fpath, sheet_name=args.sheet)
        if isinstance(x, dict):
            # multiple sheets returned
            if args.sheet and args.sheet in x:
                df = x[args.sheet]
                log.info(f"[INFO] Using sheet: {args.sheet}")
            else:
                first_sheet = list(x.keys())[0]
                df = x[first_sheet]
                log.info(f"[INFO] Using first sheet: {first_sheet}")
        else:
            df = x
    else:
        df = pd.read_csv(fpath)

    # normalize column names (but keep originals)
    df.columns = [c.strip() for c in df.columns]

    # find coordinate columns
    try:
        col_a, col_b = find_coord_columns(df)
        log.info(f"[INFO] Using coordinate columns: {col_a}, {col_b}")
    except Exception as e:
        raise ValueError(f"Failed to identify coordinate columns: {e}")

    # prepare SatelliteFetchAgent
    sat = SatelliteFetchAgent(cfg)
    sat.initialize()

    metadata_path = outdir / "metadata.jsonl"
    mf = open(metadata_path, "w", encoding="utf8")

    total = len(df)
    limit = args.limit if args.limit else total
    log.info(f"\n📸 Fetching {limit} images...\n")

    count = 0
    for idx, row in df.iterrows():
        if count >= limit:
            break
        raw_a = row[col_a]
        raw_b = row[col_b]

        lat, lon = detect_lat_lon_pair(raw_a, raw_b)
        if lat is None:
            log.warning(f"[SKIP] Row {idx}: invalid coordinate pair ({raw_a}, {raw_b})")
            continue

        sid = f"coord_{idx}"

        try:
            result = sat.run({
                "geo": {"latitude": lat, "longitude": lon},
                "submission_id": sid
            })
            src = Path(result["image_path"])
            dst = imgdir / f"{sid}.png"
            # move (or replace) into dataset folder
            src.replace(dst)

            meta = {
                "id": sid,
                "latitude": lat,
                "longitude": lon,
                "image_path": str(dst),
                "meters_per_pixel": result.get("meters_per_pixel"),
                "zoom_used": result.get("zoom_used"),
                "url": result.get("url")
            }
            mf.write(json.dumps(meta) + "\n")
            log.info(f"[OK] {sid}  (lat={lat}, lon={lon})")
            count += 1

        except Exception as e:
            log.error(f"[ERROR] Failed for row {idx}: {e}")

    mf.close()
    log.info(f"\n✅ COMPLETE — Images saved to: {imgdir}")
    log.info(f"Metadata written: {metadata_path}")

if __name__ == "__main__":
    main()
