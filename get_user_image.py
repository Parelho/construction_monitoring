import json
import glob
import time
from ultralytics import YOLO
import cv2

epi_model = YOLO("best.pt")
concrete_model = YOLO("concrete.pt")
building_model = YOLO("building.pt")
window_model = YOLO("window.pt")

while True:
    json_files = glob.glob("*.json")

    for json_file in json_files:
        with open(json_file, "r+") as file:
            data = json.load(file)

        if data["detected"] is False:

            img_path = data["input"]
            out_path = data["output"]
            img = cv2.imread(img_path)

            machine = False
            building = False
            pillar = False
            window = False

            # ============================================
            # 1) RUN EPI MODEL FIRST
            # ============================================
            epi_results = epi_model(img_path)
            epi_found = False

            for r in epi_results:
                for box in r.boxes:
                    epi_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0].item())

                    # draw only epi detections
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(img, f"epi:{cls}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # If EPI found, skip everything else
            if epi_found:
                cv2.imwrite(out_path, img)

                data["detected"] = True
                data["machine"] = False
                data["building"] = False
                data["pillar"] = False
                data["window"] = False

                with open(json_file, "w") as file:
                    json.dump(data, file, indent=4)

                continue  # SKIP OTHER MODELS

            # ============================================
            # 2) NO EPI → RUN OTHER MODELS AS NORMAL
            # ============================================

            all_boxes = []

            # concrete / pillar
            results = concrete_model(img_path)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    all_boxes.append((x1, y1, x2, y2, cls, "pillar"))
                    if cls == 0:
                        pillar = True

            # building / machine
            results = building_model(img_path)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    all_boxes.append((x1, y1, x2, y2, cls, "building_machine"))
                    if cls == 0:
                        building = True
                    elif cls == 1:
                        machine = True

            # window
            results = window_model(img_path)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    all_boxes.append((x1, y1, x2, y2, cls, "window"))
                    if cls == 0:
                        window = True

            # draw all boxes
            for (x1, y1, x2, y2, cls, src) in all_boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{src}:{cls}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imwrite(out_path, img)

            # update json
            data["machine"] = machine
            data["building"] = building
            data["pillar"] = pillar
            data["window"] = window
            data["detected"] = True

            with open(json_file, "w") as file:
                json.dump(data, file, indent=4)

    time.sleep(5)
