# training/export_seg_model.py
from ultralytics import YOLO
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="training/runs/seg_solar_panels/weights/best.pt")
parser.add_argument("--outdir", default="training/exports")
args = parser.parse_args()

weights = args.weights
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

model = YOLO(weights)

# export to ONNX and TorchScript
model.export(format="onnx", opset=11, imgsz=1024, simplify=True, output=str(outdir / "model.onnx"))
model.export(format="torchscript", imgsz=1024, output=str(outdir / "model.torchscript"))
print("Exported to:", outdir)
