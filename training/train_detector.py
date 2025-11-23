from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")  # start from pretrained
    model.train(
        data="config/solar_detector.yaml",
        imgsz=1080,
        epochs=150,
        batch=8,
        workers=4,
        device=0,          # GPU
        project="training/runs_detector",
        name="solar_detector",
        pretrained=True
    )

if __name__ == "__main__":
    main()
