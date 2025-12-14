# agents/coordinate_validation_agent.py
from .base_agent import BaseAgent

class CoordinateValidationAgent(BaseAgent):

    def initialize(self):
        return

    def run(self, payload: dict) -> dict:
        lat = payload.get("latitude")
        lon = payload.get("longitude")
        submission_id = payload.get("submission_id", "submission_auto")

        result = {
            "submission_id": submission_id,
            "geo": {
                "latitude": lat,
                "longitude": lon
            },
            "valid": True,
            "errors": []
        }

        if lat is None or lon is None:
            result["valid"] = False
            result["errors"].append("Coordinates missing.")
        else:
            if not (-90 <= lat <= 90):
                result["valid"] = False
                result["errors"].append("Latitude out of range.")

            if not (-180 <= lon <= 180):
                result["valid"] = False
                result["errors"].append("Longitude out of range.")

        return result
