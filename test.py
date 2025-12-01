from ultralytics import YOLO
import cv2

# Load the YOLO model (make sure it's a .pt model, not an image!)
model = YOLO("building.pt")

# Run inference
results = model("image.png")

# Visualize using OpenCV
for r in results:
    boxes = r.boxes
    for box in boxes:
        class_id = int(box.cls[0].item())
        print(class_id)