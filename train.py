from ultralytics import YOLO

# Load pretrained YOLO11n
model = YOLO("yolo11l.pt")

# Train with split dataset
model.train(
    data="/home/vinicius/Git/construction_monitoring/pictor.yaml",
    epochs=100,
    imgsz=640
)
