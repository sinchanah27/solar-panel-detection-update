# agents/residential_classifier_agent.py
import math
from typing import Dict, Any, List, Tuple
from .base_agent import BaseAgent

# Simple clustering by distance threshold (no sklearn required)
def cluster_centers(centers: List[Tuple[float,float]], threshold: float) -> List[List[int]]:
    """
    centers: list of (x, y)
    threshold: distance in pixels to join points in same cluster
    Returns list of clusters (each cluster is list of indices)
    """
    clusters = []
    visited = set()
    for i, c in enumerate(centers):
        if i in visited:
            continue
        cluster = [i]
        visited.add(i)
        # BFS expand
        stack = [i]
        while stack:
            idx = stack.pop()
            xi, yi = centers[idx]
            for j, cj in enumerate(centers):
                if j in visited:
                    continue
                xj, yj = cj
                dist = math.hypot(xi - xj, yi - yj)
                if dist <= threshold:
                    visited.add(j)
                    cluster.append(j)
                    stack.append(j)
        clusters.append(cluster)
    return clusters

class ResidentialClassifierAgent(BaseAgent):
    """
    Hybrid rule-based classifier: residential, apartment, industrial, solar_farm.
    Inputs expected:
      - detections: list of {"bbox":[x1,y1,x2,y2],...}
      - meters_per_pixel: float
      - roof_area_m2: float (optional; segmentation_agent output)
    Returns:
      { "classification": "residential" | "apartment" | "industrial" | "solar_farm" }
    """
    def initialize(self):
        # thresholds (can be tuned later)
        self.cluster_px_threshold = float(self.config.get("classifier", {}).get("cluster_px_threshold", 60.0))
        self.solar_farm_area_threshold_m2 = float(self.config.get("classifier", {}).get("solar_farm_area_m2", 20000.0))
        self.industrial_roof_coverage = float(self.config.get("classifier", {}).get("industrial_roof_coverage", 0.7))
        self.apartment_panel_count_threshold = int(self.config.get("classifier", {}).get("apartment_panel_count", 40))
        self.industrial_panel_count_threshold = int(self.config.get("classifier", {}).get("industrial_panel_count", 200))
        return

    def _panel_centers(self, detections):
        centers = []
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            centers.append((cx, cy))
        return centers

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        detections = payload.get("detections", [])
        mpp = payload.get("meters_per_pixel", None)
        roof_area_m2 = payload.get("roof_area_m2", None)  # may be None

        # Count panels and compute solar-covered area (internal only)
        panel_count = len(detections)
        solar_area_m2 = 0.0
        for d in detections:
            x1,y1,x2,y2 = d["bbox"]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if mpp is None:
                # fallback: treat pixel area as 1 px -> area 0
                area_m2 = 0.0
            else:
                area_m2 = (w * h) * (mpp ** 2)
            solar_area_m2 += area_m2

        # Quick rule: huge total solar area -> solar farm
        if solar_area_m2 >= self.solar_farm_area_threshold_m2:
            return {"classification": "solar_farm"}

        # roof_area unknown fallback: estimate from image size at mpp if available
        if roof_area_m2 is None:
            # attempt to derive from image shape using image_path if provided (not implemented here),
            # fallback to weak heuristic: treat roof_area as solar_area * 2 if panels exist, else 0
            if solar_area_m2 > 0:
                roof_area_m2 = max(solar_area_m2 * 2.0, solar_area_m2 + 50.0)
            else:
                roof_area_m2 = 0.0

        # safety: avoid division by zero
        roof_area_m2 = float(roof_area_m2) if roof_area_m2 is not None else 0.0

        # compute clusters from centers (pixel threshold)
        centers = self._panel_centers(detections)
        clusters = cluster_centers(centers, self.cluster_px_threshold) if centers else []

        # features
        cluster_count = len(clusters)
        max_cluster_size = max((len(c) for c in clusters), default=0)

        # Rule set (hybrid, tunable)
        # 1) industrial if roof coverage very high by solar
        if roof_area_m2 > 0 and (solar_area_m2 / roof_area_m2) >= self.industrial_roof_coverage:
            return {"classification": "industrial"}

        # 2) industrial if many panels overall
        if panel_count >= self.industrial_panel_count_threshold:
            return {"classification": "industrial"}

        # 3) apartment if moderately large panel count or big cluster
        if panel_count >= self.apartment_panel_count_threshold or max_cluster_size >= self.apartment_panel_count_threshold/2:
            return {"classification": "apartment"}

        # 4) if multiple clusters with medium sizes, classify as apartment
        if cluster_count >= 3 and panel_count > 10:
            return {"classification": "apartment"}

        # 5) default small -> residential
        return {"classification": "residential"}
