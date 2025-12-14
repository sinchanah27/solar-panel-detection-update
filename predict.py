import json
from ultralytics import YOLO

# Load model
model = YOLO("yolov8s.pt")

# Load input
with open("sample_input.json", "r") as f:
    data = json.load(f)

image_path = data["image_path"]

# Run inference
results = model(image_path)

output = {"predictions": []}

for r in results:
    for box in r.boxes:
        output["predictions"].append({
            "class": r.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": {
                "x_min": float(box.xyxy[0][0]),
                "y_min": float(box.xyxy[0][1]),
                "x_max": float(box.xyxy[0][2]),
                "y_max": float(box.xyxy[0][3])
            }
        })

# Save output
with open("sample_output.json", "w") as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
