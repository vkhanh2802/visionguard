import cv2
import numpy as np

from src.detection import Detection
from src.tracking import Track
from src.tracking.history import TrackHistory
from math import hypot
from src.events import LineCrossingEngine, IntrusionEngine, LoiteringEngine

def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        label = f"{detection.class_name} {detection.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

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

def draw_line_crossing(frame, engine: LineCrossingEngine):
    start = engine.line_start
    end = engine.line_end

    positive_start, positive_end = _offset_line(start, end, engine.dead_zone_px)
    negative_start, negative_end = _offset_line(start, end, -engine.dead_zone_px)

    cv2.line(frame, positive_start, positive_end, (100, 100, 100), 1)
    cv2.line(frame, negative_start, negative_end, (100, 100, 100), 1)
    cv2.line(frame, start, end, (0, 255, 255), 2)

    cv2.circle(frame, start, 5, (0, 255, 255), -1)
    cv2.circle(frame, end, 5, (0, 255, 255), -1)

    return frame

def _offset_line(start: tuple[int, int], end: tuple[int, int], offset: float):
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1
    length = hypot(dx, dy)

    if length == 0:
        return start, end

    nx = -dy / length
    ny = dx / length

    offset_start = (int(x1 + nx * offset), int(y1 + ny * offset))
    offset_end = (int(x2 + nx * offset), int(y2 + ny * offset))

    return offset_start, offset_end

def draw_line_directions(frame, engine: LineCrossingEngine):
    x1, y1 = engine.line_start
    x2, y2 = engine.line_end

    dx = x2 - x1
    dy = y2 - y1
    length = hypot(dx, dy)

    if length == 0:
        return frame

    nx = -dy / length
    ny = dx / length

    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2

    label_offset = 30

    positive_position = (int(mid_x + nx * label_offset), int(mid_y + ny * label_offset))
    negative_position = (int(mid_x - nx * label_offset), int(mid_y - ny * label_offset))

    cv2.putText(frame, engine.negative_to_positive, positive_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, engine.positive_to_negative, negative_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame

def draw_polygon_roi(frame: np.ndarray, polygon: tuple[tuple[float, float],...], label: str) -> np.ndarray:
    points = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        frame,
        [points],
        isClosed=True,
        color=(0, 0, 255),
        thickness=2,
    )

    x, y = points[0, 0]
    cv2.putText(
        frame,
        label,
        (int(x), max(int(y) - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return frame


def draw_event_counts( frame: np.ndarray, line_engine: LineCrossingEngine, intrusion_engine: IntrusionEngine, loitering_engine: LoiteringEngine) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2

    cv2.putText(frame, f"IN: {line_engine.in_count}", (20, 80), font, scale, (0, 255, 0), thickness)
    cv2.putText(frame, f"OUT: {line_engine.out_count}", (20, 110), font, scale, (0, 0, 255), thickness)
    cv2.putText(frame, f"INTRUSIONS: {intrusion_engine.intrusion_count}", (20, 140), font, scale, (255, 0, 255), thickness)
    cv2.putText(frame, f"LOITERING: {loitering_engine.loitering_count}", (20, 170), font, scale, (0, 165, 255), thickness)

    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return frame