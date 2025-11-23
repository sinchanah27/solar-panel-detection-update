# training/train_segmentation.py
from ultralytics import YOLO
import argparse
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="config/segmentation_dataset.yaml", help="dataset yaml")
    parser.add_argument("--model", default="yolov8s-seg.pt", help="pretrained segmentation model or yolov8s-seg")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", default="training/runs", help="output folder")
    parser.add_argument("--name", default="seg_solar_panels", help="run name")
    parser.add_argument("--device", default="0", help="cuda device or cpu")
    args = parser.parse_args()

    # ensure dataset yaml exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

    model = YOLO(args.model)  # yolov8s-seg pretrained or custom
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=4,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=False,
        seed=42
    )

if __name__ == "__main__":
    main()
