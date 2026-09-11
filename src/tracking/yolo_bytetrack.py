import numpy as np
from ultralytics import YOLO

from .types import Track

class YOLOByteTracker:
    def __init__(self, model_path="yolo26n.pt", confidence=0.4, target_classes=None):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.target_classes = set(target_classes) if target_classes else None
        self.target_class_ids = self._resolve_target_class_ids()

    def _resolve_target_class_ids(self) -> list[int] | None:
        if not self.target_classes:
            return None

        names = self.model.names
        class_ids = [
            class_id
            for class_id, class_name in names.items()
            if class_name in self.target_classes
        ]

        missing_classes = self.target_classes - {names[class_id] for class_id in class_ids}
        if missing_classes:
            available_classes = ", ".join(names.values())
            missing = ", ".join(sorted(missing_classes))
            raise ValueError(f"Unknown target class(es): {missing}. Available classes: {available_classes}")

        return class_ids

    def track(self, frame: np.ndarray) -> list[Track]:
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            classes=self.target_class_ids,
            verbose=False,
        )[0]

        if result.boxes is None:
            return []
        tracks = []
        for box in result.boxes:
            if box is None or box.id is None:
                continue
            class_id = int(box.cls[0].item())
            class_name = self.model.names[class_id]

            if self.target_classes and class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            bottom_center = ((x1 + x2) // 2, y2)
            tracks.append(
                Track(
                    track_id=int(box.id[0].item()),
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf[0].item()),
                    class_id=class_id,
                    class_name=class_name,
                    centroid=centroid,
                    bottom_center=bottom_center
                )
            )
        return tracks
