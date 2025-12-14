from ultralytics import YOLO

def main():
    model = YOLO("yolov8s-seg.pt")
    model.train(
        data="config/solar_segmentation.yaml",
        imgsz=1080,
        epochs=200,
        batch=6,
        workers=4,
        device=0,
        project="training/runs_segment",
        name="solar_seg",
        pretrained=True
    )

if __name__ == "__main__":
    main()
