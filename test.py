from ultralytics import YOLO
import os

# Load your trained model
model = YOLO(r"venv\runs\detect\train\weights\best.pt")

# Folder with images to evaluate
images_dir = r"construction\test\images"

# Output folder for detections
output_dir = "detections"
os.makedirs(output_dir, exist_ok=True)

# Run detection and save results per image
results = model.predict(
    source=images_dir,       # directory or single image/video path
    save=True,               # save annotated images
    project=output_dir,      # parent folder
    name="",                 # optional subfolder name
    save_txt=True,           # save results as YOLO-format text files
    save_conf=True           # save confidence values
)

print("✅ Detections saved in:", output_dir)
