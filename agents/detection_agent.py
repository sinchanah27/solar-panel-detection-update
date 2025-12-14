# agents/detection_agent.py
from typing import Dict, Any
from .base_agent import BaseAgent
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import torch

class DetectionAgent(BaseAgent):
    """
    YOLOv8 Detection Agent
    - Loads YOLOv8 model (detector or seg model)
    - Returns bounding boxes and confidence scores
    - Integrates with meters_per_pixel + zoom_used
    """

    def initialize(self):
        # Load YOLO model path from config
        self.model_path = (
            self.config.get("yolo", {}).get("model_path", None)
        )

        if not self.model_path:
            raise ValueError("YOLO model_path missing in config['yolo']['model_path']")

        # Load YOLO once (important for speed)
        self.model = YOLO(self.model_path)

        # Confidence threshold
        self.conf_thres = float(self.config.get("yolo", {}).get("conf", 0.25))

        # Whether to refine detections with segmentation masks
        self.use_seg_refine = bool(
            self.config.get("segmentation", {}).get("use_seg_refine", False)
        )

    def _yolo_detect(self, image_path: str):
        """Run YOLOv8 inference & extract bboxes."""

        # YOLO predict
        results = self.model.predict(
            source=image_path,
            conf=self.conf_thres,
            verbose=False
        )

        r = results[0]  # YOLO result for this image

        detections = []

        if r.boxes is None:
            return detections

        for box in r.boxes:
            # xyxy format
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf
            })

        return detections

    def _refine_with_segmentation(self, image_path: str, detections, payload):
        """Optional refinement using segmentation mask."""
        try:
            from .segmentation_agent import SegmentationAgent

            seg = SegmentationAgent(self.config)
            seg.initialize()

            seg_out = seg.run({
                "image_path": image_path,
                "meters_per_pixel": payload.get("meters_per_pixel"),
                "submission_id": payload.get("submission_id")
            })

            mask = cv2.imread(seg_out["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return detections  # no refinement possible

            refined = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]

                x1i = max(0, x1)
                y1i = max(0, y1)
                x2i = min(mask.shape[1]-1, x2)
                y2i = min(mask.shape[0]-1, y2)

                if x2i <= x1i or y2i <= y1i:
                    continue

                roi = mask[y1i:y2i, x1i:x2i]
                if roi.size == 0:
                    continue

                overlap = np.count_nonzero(roi) / roi.size

                if overlap >= 0.1:   # require 10% mask overlap
                    refined.append(d)

            return refined if len(refined) > 0 else detections

        except Exception as e:
            print(f"[WARN] Segmentation refine failed: {e}")
            return detections

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image_path = payload.get("image_path")
        if not image_path:
            raise ValueError("DetectionAgent: image_path missing")

        # Step 1: YOLO detection
        detections = self._yolo_detect(image_path)

        # Step 2: OPTIONAL segmentation refinement
        if self.use_seg_refine:
            detections = self._refine_with_segmentation(
                image_path,
                detections,
                payload
            )

        # Return detections with calibration metadata
        return {
            "detections": detections,
            "image_path": image_path,
            "meters_per_pixel": payload.get("meters_per_pixel"),
            "zoom_used": payload.get("zoom_used")
        }
