from dataclasses import dataclass

@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float 
    class_id : int
    class_name: str