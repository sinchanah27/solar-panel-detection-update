# agents/segmentation_agent.py
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Any
from .base_agent import BaseAgent

class SegmentationAgent(BaseAgent):
    """
    ML-based segmentation agent
    Uses YOLOv8-Seg model to produce panel masks for area estimation.
    """

    def initialize(self):
        self.model_path = self.config.get("segmentation", {}).get("model_path", None)

        if not self.model_path:
            raise ValueError("SegmentationAgent: segmentation.model_path missing in config")

        self.model = YOLO(self.model_path)

        self.conf_thres = float(
            self.config.get("segmentation", {}).get("conf", 0.25)
        )

        self.temp_dir = Path("data/segmentation_masks")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = payload["image_path"]
        submission_id = payload.get("submission_id")

        results = self.model.predict(
            source=image_path,
            conf=self.conf_thres,
            verbose=False
        )

        r = results[0]

        # save a single combined binary mask
        h, w = r.orig_img.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        mask_count = 0
        mask_paths = []

        if r.masks is not None:
            for i, m in enumerate(r.masks.data):
                mask = m.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)

                # upsample to original resolution
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                combined_mask[mask > 0] = 255

                # individual mask file (optional)
                out_path = self.temp_dir / f"{submission_id}_mask_{i}.png"
                cv2.imwrite(str(out_path), mask)
                mask_paths.append(str(out_path))
                mask_count += 1

        # save combined mask
        combined_path = self.temp_dir / f"{submission_id}_mask_combined.png"
        cv2.imwrite(str(combined_path), combined_mask)

        return {
            "combined_mask_path": str(combined_path),
            "individual_masks": mask_paths,
            "mask_count": mask_count
        }
