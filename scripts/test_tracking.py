import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture("data/videos/test.mp4")

if not cap.isOpened():
    raise RuntimeError("Cannot open video")

while True:
    success, frame = cap.read()

    if not success:
        break

    result = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
    annotated_frame = result.plot()

    cv2.imshow("VisionGuard Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()