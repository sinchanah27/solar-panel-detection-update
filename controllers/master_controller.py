# controllers/master_controller.py
import argparse
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

# Agents
from agents.coordinate_validation_agent import CoordinateValidationAgent
from agents.satellite_fetch_agent import SatelliteFetchAgent
from agents.detection_agent import DetectionAgent
from agents.segmentation_agent import SegmentationAgent
from agents.quantification_agent import QuantificationAgent
from agents.audit_qc_agent import AuditQCAgent

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_config(path="config/config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



def run_pipeline(payload: dict, config: dict):
    """
    Full ML pipeline:
      1. Coordinate Validation
      2. Satellite Fetch
      3. YOLO Detection
      4. YOLO-Seg Segmentation
      5. ML Area Estimation
      6. Audit QC (final JSON + overlay output)
    """

    submission_id = payload["submission_id"]
    lat = payload["latitude"]
    lon = payload["longitude"]

    logging.info(f"Starting pipeline for submission_id={submission_id}")

    # ---------------------------------------------------------------------
    # 1. Coordinate validation
    # ---------------------------------------------------------------------
    logging.info("Stage 1: Validate coordinates")
    coord = CoordinateValidationAgent(config)
    coord.initialize()
    coord_out = coord.run(payload)

    if not coord_out["valid"]:
        return {
            "submission_id": submission_id,
            "error": "INVALID_COORDS",
            "details": coord_out
        }

    # ---------------------------------------------------------------------
    # 2. Satellite image fetch (small-area zoom first)
    # ---------------------------------------------------------------------
    logging.info("Stage 2: Fetch satellite image (zoom_small)")
    sat = SatelliteFetchAgent(config)
    sat.initialize()

    sat_payload = {
        "submission_id": submission_id,
        "geo": coord_out["geo"],
        "stage": "small"
    }

    try:
        sat_out = sat.run(sat_payload)
    except Exception as e:
        return {
            "submission_id": submission_id,
            "error": "SATELLITE_FETCH_FAILED",
            "details": str(e)
        }

    image_path = sat_out["image_path"]
    meters_per_pixel = sat_out.get("meters_per_pixel")
    zoom_used = sat_out.get("zoom_used")

    # ---------------------------------------------------------------------
    # 3. YOLO detection
    # ---------------------------------------------------------------------
    logging.info("Stage 3: YOLO detection")
    det = DetectionAgent(config)
    det.initialize()

    det_out = det.run({
        "image_path": image_path,
        "meters_per_pixel": meters_per_pixel,
        "zoom_used": zoom_used,
        "submission_id": submission_id
    })

    # ---------------------------------------------------------------------
    # 4. YOLO segmentation
    # ---------------------------------------------------------------------
    logging.info("Stage 4: YOLO-Seg segmentation (ML masks)")
    seg = SegmentationAgent(config)
    seg.initialize()

    seg_out = seg.run({
        "image_path": image_path,
        "meters_per_pixel": meters_per_pixel,
        "submission_id": submission_id
    })

    # ---------------------------------------------------------------------
    # 5. ML area estimation
    # ---------------------------------------------------------------------
    logging.info("Stage 5: ML area estimation")
    quant = QuantificationAgent(config)
    quant.initialize()

    quant_out = quant.run({
        "detections": det_out["detections"],
        "segmentation_result": seg_out,
        "meters_per_pixel": meters_per_pixel,
        "zoom_used": zoom_used
    })

    # ---------------------------------------------------------------------
    # 6. Audit & QC – FINAL OUTPUT
    # ---------------------------------------------------------------------
    logging.info("Stage 6: Audit & QC")

    qc = AuditQCAgent(config)
    qc.initialize()

    qc_payload = {
        "submission_id": submission_id,
        "geo": coord_out["geo"],

        # REQUIRED: missing earlier — now fixed
        "image_path": image_path,
        "meters_per_pixel": meters_per_pixel,
        "zoom_used": zoom_used,

        # ML results passed in
        "detection": quant_out,
        "segmentation_result": seg_out
    }

    qc_out = qc.run(qc_payload)

    # Build final structure returned to CLI or batch runner
    final = {
        "submission_id": submission_id,
        "geo": coord_out["geo"],

        "detection": quant_out,
        "qc": qc_out,

        "meta": {
            "created_at": datetime.utcnow().isoformat() + "Z"
        },

        "output_path": qc_out.get("output_path")
    }

    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--submission_id", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    payload = {
        "latitude": args.lat,
        "longitude": args.lon,
        "submission_id": args.submission_id
    }

    result = run_pipeline(payload, cfg)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
