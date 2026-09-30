from pathlib import Path

import numpy as np

from src.tracking.yolo_bytetrack import YOLOByteTracker


class FakeBox:
    id = None


class FakeResult:
    boxes = [FakeBox()]


class FakeYOLO:
    def __init__(self, _model_path: str):
        self.names = {0: "person"}
        self.track_kwargs = None

    def track(self, _frame: np.ndarray, **kwargs):
        self.track_kwargs = kwargs
        return [FakeResult()]


class FakeTracker:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def test_uses_custom_tracker_config(monkeypatch):
    monkeypatch.setattr("src.tracking.yolo_bytetrack.YOLO", FakeYOLO)
    tracker_config = Path("configs/bytetrack_camera.yaml")
    tracker = YOLOByteTracker(tracker_config=tracker_config)

    assert tracker.track(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    assert tracker.model.track_kwargs["tracker"] == str(tracker_config)


def test_uses_ultralytics_bytetrack_default_without_custom_config(monkeypatch):
    monkeypatch.setattr("src.tracking.yolo_bytetrack.YOLO", FakeYOLO)
    tracker = YOLOByteTracker()

    tracker.track(np.zeros((10, 10, 3), dtype=np.uint8))

    assert tracker.model.track_kwargs["tracker"] == "bytetrack.yaml"


def test_resets_persistent_ultralytics_tracker(monkeypatch):
    monkeypatch.setattr("src.tracking.yolo_bytetrack.YOLO", FakeYOLO)
    tracker = YOLOByteTracker()
    byte_tracker = FakeTracker()
    tracker.model.predictor = type(
        "FakePredictor",
        (),
        {"trackers": [byte_tracker], "vid_path": ["previous.mp4"]},
    )()

    tracker.reset()

    assert byte_tracker.reset_calls == 1
    assert tracker.model.predictor.vid_path == [None]
