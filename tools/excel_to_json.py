#!/usr/bin/env python3
"""
Advanced batch runner: Excel/CSV -> pipeline -> JSON outputs + summary report.
"""

import argparse
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import time

from controllers.master_controller import load_config, run_pipeline


# -------------------------
# Validation helpers
# -------------------------

def is_valid_lat(lat):
    try:
        return -90.0 <= float(lat) <= 90.0
    except:
        return False


def is_valid_lon(lon):
    try:
        return -180.0 <= float(lon) <= 180.0
    except:
        return False


def make_submission_id(prefix: str, idx: int) -> str:
    ts = int(time.time())
    return f"{prefix}_{ts}_{idx}"


# -------------------------
# Run one record
# -------------------------

def run_one(record: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Run pipeline for a single Excel row."""

    try:
        lat = float(record["latitude"])
        lon = float(record["longitude"])
    except Exception as e:
        return {"status": "invalid_input", "error": f"Invalid lat/lon: {e}", "row": record}

    submission_id = record.get("submission_id")
    if not submission_id or str(submission_id).strip() == "":
        submission_id = make_submission_id("auto", record.get("_idx", 0))

    payload = {"latitude": lat, "longitude": lon, "submission_id": submission_id}

    try:
        result = run_pipeline(payload, config)

        if result.get("error"):  # pipeline returned an error
            return {
                "status": "pipeline_error",
                "submission_id": submission_id,
                "error": result.get("error"),
                "details": result
            }

        qc = result.get("qc", {})
        return {
            "status": "success",
            "submission_id": submission_id,
            "output_path": result.get("output_path"),
            "qc_status": qc.get("status"),
            "qc_reason_codes": qc.get("reason_codes", [])
        }

    except Exception as e:
        return {
            "status": "exception",
            "submission_id": submission_id,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--outdir", "-o", default="data/outputs")
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parser.add_argument("--workers", "-w", type=int, default=1)
    parser.add_argument("--write_xlsx", action="store_true")
    parser.add_argument("--prefix", default="submission")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load file (FIXED SECTION)
    # -------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() in [".xls", ".xlsx"]:
        if args.sheet is None:
            df = pd.read_excel(input_path, sheet_name=0)
        else:
            df = pd.read_excel(input_path, sheet_name=args.sheet)

    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)

    else:
        raise ValueError("Unsupported file type. Use .xlsx/.xls or .csv")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate columns
    required_cols = ["latitude", "longitude"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # prepare record list
    records = []
    for idx, row in df.iterrows():
        r = {col: row[col] for col in df.columns}
        r["_idx"] = idx
        if "submission_id" in r and (pd.isna(r["submission_id"]) or str(r["submission_id"]).strip() == ""):
            r.pop("submission_id", None)
        records.append(r)

    # -------------------------
    # Run pipeline batch
    # -------------------------
    summary = {
        "total_rows": len(records),
        "processed": 0,
        "successes": 0,
        "failures": 0,
        "items": []
    }

    def worker(rec):
        idx = rec["_idx"]
        if not is_valid_lat(rec["latitude"]) or not is_valid_lon(rec["longitude"]):
            return {
                "status": "invalid_coords",
                "row_index": idx,
                "latitude": rec["latitude"],
                "longitude": rec["longitude"]
            }

        if "submission_id" not in rec:
            rec["submission_id"] = make_submission_id(args.prefix, idx)

        return run_one(rec, cfg)

    results = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(worker, r): r for r in records}
            for f in as_completed(futures):
                results.append(f.result())
    else:
        for r in records:
            results.append(worker(r))

    # -------------------------
    # Collect summary
    # -------------------------
    for r in results:
        summary["processed"] += 1
        if r.get("status") == "success":
            summary["successes"] += 1
        else:
            summary["failures"] += 1
        summary["items"].append(r)

    # Save summary JSON
    summary_path = outdir / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Batch complete")
    print(f"Summary written: {summary_path}")
    print(f"Success: {summary['successes']} | Fail: {summary['failures']}")


if __name__ == "__main__":
    main()
