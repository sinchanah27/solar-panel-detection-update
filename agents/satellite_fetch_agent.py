# agents/satellite_fetch_agent.py

import os
import time
import math
import requests
from pathlib import Path
from typing import Dict, Any
from .base_agent import BaseAgent
from urllib.parse import quote_plus


class SatelliteFetchAgent(BaseAgent):
    """
    Fetch high-quality Mapbox satellite tiles (1280x1280, zoom 19).
    Automatically retries zoom levels 19 → 18 → 17 to prevent black images.
    """

    def initialize(self):
        self.token = os.getenv("MAPBOX_TOKEN") or self.config.get("satellite", {}).get("mapbox_token")
        if not self.token:
            raise RuntimeError("MAPBOX_TOKEN not found. Set it in environment variables.")

        # Mapbox maximum size is 1280×1280
        self.size = 1280

        # Default zoom
        self.default_zoom = int(self.config.get("satellite", {}).get("zoom_default", 19))

        # Default output directory
        self.outdir = Path(self.config.get("output", {}).get("outdir", "data/outputs"))
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self.last_call = 0.0
        self.min_interval = 0.2

    @staticmethod
    def mpp(lat: float, zoom: int):
        """Meters per pixel (Web Mercator)."""
        return 156543.03392 * math.cos(lat * math.pi / 180) / (2 ** zoom)

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Rate limit
        now = time.time()
        if now - self.last_call < self.min_interval:
            time.sleep(self.min_interval - (now - self.last_call))
        self.last_call = time.time()

        lat = float(payload["geo"]["latitude"])
        lon = float(payload["geo"]["longitude"])
        submission_id = payload["submission_id"]

        # Zoom attempts 19 → 18 → 17
        zoom_attempts = [self.default_zoom, 18, 17]

        outfile = self.outdir / f"{submission_id}_sat.png"

        for attempt, zoom in enumerate(zoom_attempts, start=1):
            size_str = f"{self.size}x{self.size}"

            url = (
                f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
                f"{lon},{lat},{zoom}/{size_str}@2x"
                f"?access_token={quote_plus(self.token)}"
            )

            try:
                resp = requests.get(url, stream=True, timeout=40)
                resp.raise_for_status()

                # Save tile
                with open(outfile, "wb") as f:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            f.write(chunk)

                # Success → return result
                return {
                    "image_path": str(outfile),
                    "provider": "mapbox",
                    "url": url,
                    "meters_per_pixel": self.mpp(lat, zoom),
                    "zoom_used": zoom
                }

            except Exception:
                if attempt == len(zoom_attempts):
                    raise RuntimeError(f"Tile fetch failed for {submission_id}")
                time.sleep(0.4)  # wait before retry

