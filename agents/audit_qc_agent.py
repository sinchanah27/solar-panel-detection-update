# agents/audit_qc_agent.py
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
from .base_agent import BaseAgent
from datetime import datetime

# constants
FT2_TO_M2 = 0.09290304

def sqft_to_radius_m(sqft: float) -> float:
    """Convert area in sqft to radius in meters for a circle of that area."""
    area_m2 = sqft * FT2_TO_M2
    radius_m = math.sqrt(area_m2 / math.pi)
    return radius_m

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

class AuditQCAgent(BaseAgent):
    """
    Audit / QC agent that:
     - applies buffer logic (1200 sqft -> 2400 sqft fallback)
     - computes overlap of per-panel masks with buffer
     - picks largest overlapping panel area (m2)
     - writes overlay PNG for explainability
     - writes final JSON that matches challenge schema
    """

    def initialize(self):
        # output directory
        self.outdir = Path(self.config.get("output", {}).get("outdir", "data/outputs"))
        ensure_dir(self.outdir)
        # overlay folder (inside outdir)
        self.overlay_dir = self.outdir / "overlays"
        ensure_dir(self.overlay_dir)
        # segmentation temp dir fallback
        self.seg_temp_dir = Path(self.config.get("segmentation", {}).get("temp_dir", "data/segmentation_masks"))
        ensure_dir(self.seg_temp_dir)
        # thresholds
        self.min_overlap_pixels = int(self.config.get("qc", {}).get("min_overlap_pixels", 1))
        return

    def _make_buffer_mask(self, image_shape, meters_per_pixel, radius_m):
        """
        Create a binary circular mask (uint8) same size as image.
        image_shape: (h,w)
        meters_per_pixel: float
        radius_m: float
        """
        h, w = image_shape[:2]
        # center pixel is center of image (satellite fetch centered at coord)
        cx = w // 2
        cy = h // 2
        radius_px = int(round(radius_m / meters_per_pixel))
        buf_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(buf_mask, (cx, cy), radius_px, 255, thickness=-1)
        return buf_mask, (cx, cy, radius_px)

    def _read_mask(self, path: str, expected_shape):
        """
        Read grayscale mask. If shape differs, resize to expected_shape using nearest.
        """
        if not path or not Path(path).exists():
            return np.zeros(expected_shape[:2], dtype=np.uint8)
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.zeros(expected_shape[:2], dtype=np.uint8)
        if (m.shape[0], m.shape[1]) != (expected_shape[0], expected_shape[1]):
            m = cv2.resize(m, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
        # ensure binary
        _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
        return m

    def _compute_overlap_area_m2(self, mask_path: str, buffer_mask: np.ndarray, meters_per_pixel: float):
        """
        Compute overlap area in m2 between panel mask and buffer mask.
        Returns overlap_pixels, overlap_m2
        """
        mask = self._read_mask(mask_path, buffer_mask.shape + (3,))  # trick expected_shape
        overlap = cv2.bitwise_and(mask, buffer_mask)
        overlap_pixels = int(np.count_nonzero(overlap))
        area_m2 = float(overlap_pixels * (meters_per_pixel ** 2))
        return overlap_pixels, area_m2

    def _draw_overlay(self, image_path: str, combined_mask_path: Optional[str], detections, buf_center, out_path: Path, pv_area_m2: float, confidence: float, buffer_sqft: int):
        """
        Draw overlay PNG with:
         - background image
         - buffer circle
         - combined mask contours
         - detection boxes
         - textual annotations
        """
        img = cv2.imread(image_path)
        if img is None:
            # create blank image if missing
            img = np.zeros((1080,1080,3), dtype=np.uint8)
        overlay = img.copy()
        h, w = overlay.shape[:2]

        cx, cy, radius_px = buf_center

        # draw buffer circle (green)
        cv2.circle(overlay, (cx, cy), radius_px, (0, 255, 0), 2)

        # draw combined mask contours in blue
        if combined_mask_path and Path(combined_mask_path).exists():
            cmask = self._read_mask(combined_mask_path, overlay.shape)
            contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)

        # draw bounding boxes (if any) in orange
        for d in detections:
            bbox = d.get("bbox")
            conf = d.get("confidence", 0.0)
            if bbox:
                x1,y1,x2,y2 = bbox
                cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,140,255), 2)
                cv2.putText(overlay, f"{conf:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,140,255), 1, cv2.LINE_AA)

        # text summary box (bottom-left)
        text1 = f"pv_area_sqm_est: {pv_area_m2:.3f} m2"
        text2 = f"confidence: {confidence:.3f} | buffer: {buffer_sqft} sqft"
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        cv2.rectangle(overlay, (5, h-70), (w-5, h-5), (0,0,0), -1)
        cv2.putText(overlay, text1, (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, text2, (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"timestamp: {now}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        cv2.imwrite(str(out_path), overlay)
        return str(out_path)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects payload:
            submission_id
            geo: {latitude, longitude}
            image_path
            meters_per_pixel
            zoom_used
            detection: quant_out (with 'detections' list)  -- each detection may have 'mask_path' or bbox & confidence
            segmentation_result: seg_out (contains 'combined_mask_path' and 'individual_masks')
        Produces final JSON and writes overlay image.
        """
        try:
            submission_id = payload.get("submission_id")
            geo = payload.get("geo", {})
            lat = geo.get("latitude")
            lon = geo.get("longitude")
            image_path = payload.get("image_path")
            mpp = payload.get("meters_per_pixel")
            zoom_used = payload.get("zoom_used")
            quant = payload.get("detection", {}) or {}
            seg = payload.get("segmentation_result", {}) or {}

            # defaults and safety
            if not submission_id:
                submission_id = payload.get("submission_id", "unknown")
            if mpp is None:
                raise ValueError("meters_per_pixel missing for AuditQCAgent")

            # read image to get shape
            img = cv2.imread(image_path)
            if img is None:
                # fallback sized image (won't be ideal)
                img = np.zeros((1080,1080,3), dtype=np.uint8)
            h, w = img.shape[:2]

            # segmentation combined mask path and individual masks
            combined_mask_path = seg.get("combined_mask_path")
            individual_masks = seg.get("individual_masks", [])

            # detection list (from quant) — use mask_path if available, else bbox
            detections = quant.get("detections", [])

            # buffer radii sqft options
            candidates = [1200, 2400]
            chosen_buffer = None
            chosen_overlap_m2 = 0.0
            chosen_overlap_pixels = 0
            chosen_mask_path = None
            chosenmask_index = None
            chosen_confidence = 0.0

            # attempt each buffer radius (small first)
            for budget in candidates:
                radius_m = sqft_to_radius_m(float(budget))
                buf_mask, buf_center = self._make_buffer_mask(img.shape, mpp, radius_m)

                # compute overlap for each individual mask if available
                best_pixels = 0
                best_area_m2 = 0.0
                best_mask = None
                best_idx = None
                best_conf = 0.0

                # If individual masks from seg are available, use those
                if individual_masks:
                    for idx, mp in enumerate(individual_masks):
                        overlap_pixels, overlap_m2 = self._compute_overlap_area_m2(mp, buf_mask, mpp)
                        if overlap_pixels > best_pixels:
                            best_pixels = overlap_pixels
                            best_area_m2 = overlap_m2
                            best_mask = mp
                            best_idx = idx
                            # find corresponding confidence if detection list includes per-mask confidence
                            # detection ordering may not match masks; fallback: use max conf from detections
                    # fallback for confidence
                    if best_pixels > 0:
                        # try to derive per-panel confidence from detections by selecting max conf available
                        confs = [d.get("confidence", 0.0) for d in detections if d.get("confidence") is not None]
                        best_conf = max(confs) if confs else 0.0

                else:
                    # If no individual masks, but quant has bbox entries, approximate overlap by computing bbox center inside buffer
                    for idx, d in enumerate(detections):
                        bbox = d.get("bbox")
                        if not bbox:
                            continue
                        x1,y1,x2,y2 = bbox
                        # clamp
                        x1i = max(0, int(x1)); y1i = max(0, int(y1)); x2i = min(w-1, int(x2)); y2i = min(h-1, int(y2))
                        if x2i <= x1i or y2i <= y1i:
                            continue
                        # center
                        cx = int((x1i + x2i) / 2)
                        cy = int((y1i + y2i) / 2)
                        # check if center within buffer
                        if buf_mask[cy, cx] > 0:
                            # approximate area using bbox pixel area
                            bbox_pixels = max(0, (x2i - x1i) * (y2i - y1i))
                            area_m2 = float(bbox_pixels * (mpp ** 2))
                            if bbox_pixels > best_pixels:
                                best_pixels = bbox_pixels
                                best_area_m2 = area_m2
                                best_mask = None
                                best_idx = idx
                                best_conf = d.get("confidence", 0.0)

                if best_pixels >= self.min_overlap_pixels:
                    chosen_buffer = budget
                    chosen_overlap_pixels = int(best_pixels)
                    chosen_overlap_m2 = float(best_area_m2)
                    chosen_mask_path = best_mask
                    chosenmask_index = best_idx
                    chosen_confidence = float(best_conf)
                    # stop at first buffer that yields panels (small preferred)
                    break

            # decide has_solar and qc_status
            has_solar = bool(chosen_overlap_pixels and chosen_overlap_pixels > 0)
            qc_status = "VERIFIABLE" if has_solar else "NOT_VERIFIABLE"

            # prepare bbox_or_mask: prefer mask path if available, else use combined mask path, else null
            bbox_or_mask = chosen_mask_path or combined_mask_path or None

            # image_metadata
            image_metadata = {
                "provider": self.config.get("satellite", {}).get("provider", "mapbox"),
                "zoom_used": zoom_used,
                "meters_per_pixel": mpp,
                "capture_date": None
            }

            # overlay image
            overlay_path = self.overlay_dir / f"{submission_id}_overlay.png"
            # compute buffer center (we already have from last tested buf_center if exists, else create for 1200)
            if 'buf_center' not in locals():
                # create for default 1200
                radius_m = sqft_to_radius_m(1200.0)
                _, buf_center = self._make_buffer_mask(img.shape, mpp, radius_m)
            try:
                self._draw_overlay(image_path, combined_mask_path, detections, buf_center, overlay_path, chosen_overlap_m2, chosen_confidence, chosen_buffer or 1200)
            except Exception:
                # ignore overlay failures but capture path
                pass

            # assemble final JSON in required format
            out = {
                "sample_id": submission_id,
                "lat": lat,
                "lon": lon,
                "has_solar": 1 if has_solar else 0,
                "confidence": round(chosen_confidence, 4),
                "pv_area_sqm_est": round(chosen_overlap_m2, 3),
                "buffer_radius_sqft": int(chosen_buffer or 1200),
                "qc_status": qc_status,
                "bbox_or_mask": bbox_or_mask,
                "image_metadata": image_metadata,
                "overlay_path": str(overlay_path),
                "meta": {
                    "created_at": datetime.utcnow().isoformat() + "Z"
                }
            }

            # write out JSON file
            out_path = self.outdir / f"{submission_id}.json"
            with open(out_path, "w", encoding="utf8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

            # also return object (useful for CLI)
            out["output_path"] = str(out_path)
            return out

        except Exception as e:
            # return error structure
            return {"error": str(e)}
