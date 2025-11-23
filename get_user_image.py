import json
import glob
import time
from ultralytics import YOLO

model = YOLO("best.pt")

while True:
    json_files = glob.glob("*.json")

    for json_file in json_files:
        with open(json_file, "r+") as file:
            data = json.load(file)

            if data["detected"] == False:
                result = model(data["input"])[0]

                conf_thresh = 0.60
                keep = result.boxes.conf >= conf_thresh
                result.boxes = result.boxes[keep]

                result.save(filename=data["output"])

                data["detected"] = True

                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

    time.sleep(5)