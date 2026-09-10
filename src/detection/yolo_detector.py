import numpy as np
from ultralytics import YOLO

from.types import Detection

class YOLODetector:
    def __init__(self, model_path: str = "yolo26n.pt", confidence: float = 0.4, target_classes: set[str] | None = None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_classes = target_classes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            if self.target_classes and class_name not in self.target_classes:
                continue

            bbox = tuple(box.xyxy[0].cpu().numpy().astype(int))
            detections.append(Detection(bbox=bbox, confidence=float(box.conf[0]), class_id=class_id, class_name=class_name))

        return detections