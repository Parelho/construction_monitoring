from ultralytics import YOLO

model = YOLO(r"venv\runs\detect\train\weights\best.pt")

model.val(data="construction\data.yaml")