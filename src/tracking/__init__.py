from .types import Track

__all__ = ["Track", "YOLOByteTracker"]


def __getattr__(name: str):
    if name == "YOLOByteTracker":
        from .yolo_bytetrack import YOLOByteTracker

        return YOLOByteTracker

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")