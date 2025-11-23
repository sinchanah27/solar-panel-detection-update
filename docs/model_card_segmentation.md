# Model Card — Solar Panel Segmentation (YOLOv8-seg)

**Model name:** solar-panel-segmentation-v0

**Purpose:** Pixel-wise segmentation of rooftop solar panels from satellite imagery to compute per-panel area (m²) for PM Surya Ghar verification.

**Architecture:** YOLOv8 segmentation (yolov8s-seg recommended start).

**Input:** Satellite image (1080×1080 or resized to training imgsz).

**Output:** Binary mask of solar panels (uint8 PNG), and panel_area_m2 estimate.

**Training data:** To be provided by team — images + pixel masks. Ensure diversity: houses, apartments, industrial, different roof colors, seasons, shadows.

**Evaluation metrics:** mAP (segmentation), IoU, pixel F1 (precision/recall), and area error (m² relative error).

**Bias & Limitations:** May underperform on heavily shaded roofs, small/oblique panels, low-resolution imagery, or novel roof materials.

**Intended use:** PM Surya Ghar rooftop verification. Not to be used for surveillance or non-consensual monitoring.

**Versioning & reproducibility:** Store training logs, config, and seed. Use deterministic augmentation where possible.
