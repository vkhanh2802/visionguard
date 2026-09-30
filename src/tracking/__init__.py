from .continuity import TrackContinuityManager
from .deduplication import ActiveTrackDeduplicator
from .types import Track

__all__ = [
    "ActiveTrackDeduplicator",
    "Track",
    "TrackContinuityManager",
    "YOLOByteTracker",
]


def __getattr__(name: str):
    if name == "YOLOByteTracker":
        from .yolo_bytetrack import YOLOByteTracker

        return YOLOByteTracker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
