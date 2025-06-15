from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("global_model_checkpoints/global_model_r5.pt")
classNames = model.names

image_path = "C:/Users/srini/OneDrive/Desktop/centralized learning/testimages/img3.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
else:
    img = cv2.resize(img, [720, 640], interpolation=cv2.INTER_AREA)
    results = model(img, show=False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil(box.conf[0] * 100) / 100
            cls = int(box.cls[0])

            print(f"Class: {classNames[cls]}, Confidence: {conf}, BBox: ({x1}, {y1}, {w}, {h})")

            cvzone.putTextRect(img, f'{classNames[cls]}{","} {conf}',
                               pos=(max(0, x1), max(35, y1 - 10)),
                               scale=1, thickness=2, offset=4, colorT=(255, 0, 0), colorR=(255, 200, 200))

    cv2.imshow("Image with Detections", img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
