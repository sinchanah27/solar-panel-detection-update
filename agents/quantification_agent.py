# agents/quantification_agent.py

from typing import Dict, Any
from .base_agent import BaseAgent
import numpy as np
import cv2

class QuantificationAgent(BaseAgent):
    """
    ML-based panel area estimator using segmentation masks.
    Computes area = (#mask_pixels) * (meters_per_pixel^2)
    Does NOT return summed total area — strictly per-panel.
    """

    def initialize(self):
        return

    def _mask_area_m2(self, mask_path: str, mpp: float):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            return 0.0

        pixel_count = np.count_nonzero(mask)

        return float(pixel_count * (mpp ** 2))

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        mpp = payload.get("meters_per_pixel")
        zoom_used = payload.get("zoom_used")

        seg = payload.get("segmentation_result", None)

        if seg is None:
            raise ValueError(
                "QuantificationAgent: segmentation_result missing (YOLO-Seg output required)"
            )

        mask_paths = seg.get("individual_masks", [])
        combined_mask = seg.get("combined_mask_path")

        outputs = []

        for mpath in mask_paths:
            area_m2 = self._mask_area_m2(mpath, mpp)

            outputs.append({
                "mask_path": mpath,
                "area_m2": round(area_m2, 3)
            })

        return {
            "panels_detected": len(outputs),
            "detections": outputs,
            "meters_per_pixel": mpp,
            "zoom_used": zoom_used,
            "combined_mask": combined_mask
        }
