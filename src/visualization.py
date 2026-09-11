import cv2
import numpy as np

from src.detection import Detection
from src.tracking import Track
from src.tracking.history import TrackHistory

def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame

def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame

def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        label = f"{track.class_name} #{track.track_id}: {track.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame

def draw_trajectories(frame: np.ndarray, tracks: list[Track], history: TrackHistory) -> np.ndarray:
    for track in tracks:
        points = history.get(track.track_id)

        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (255, 255, 0), 2)

    return frame