from dataclasses import dataclass

@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    centroid: tuple[int, int]  # (cx, cy)
    bottom_center: tuple[int, int]  # (bx, by)