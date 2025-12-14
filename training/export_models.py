from ultralytics import YOLO

def export_detector():
    model = YOLO("training/runs_detector/solar_detector/weights/best.pt")
    model.export(format="onnx")
    model.export(format="torchscript")

def export_segment():
    model = YOLO("training/runs_segment/solar_seg/weights/best.pt")
    model.export(format="onnx")
    model.export(format="torchscript")

if __name__ == "__main__":
    export_detector()
    export_segment()
