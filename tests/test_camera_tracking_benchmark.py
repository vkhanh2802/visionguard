import hashlib
from argparse import Namespace

import pytest

from scripts.evaluate_camera_tracking import (
    build_run_specs,
    collect_reproducibility_metadata,
    describe_artifact,
    resolve_tracker_config,
    run_tracking,
    select_best_runs,
)
from scripts.prepare_camera_ground_truth import normalize_mot_rows


def test_normalizes_boxes_to_video_bounds():
    rows = [
        ["1", "3", "-2", "8", "12", "15", "1", "1", "1"],
        ["2", "3", "90", "95", "20", "10", "1", "1", "0"],
    ]

    normalized, clipped = normalize_mot_rows(rows, 2, 100, 100)

    assert clipped == 2
    assert normalized == [
        "1,3,0.00,8.00,10.00,15.00,1,1,1\n",
        "2,3,90.00,95.00,10.00,5.00,1,1,0\n",
    ]


def test_rejects_duplicate_frame_track_pair():
    rows = [
        ["1", "3", "0", "0", "10", "10", "1", "1", "1"],
        ["1", "3", "1", "1", "10", "10", "1", "1", "1"],
    ]

    with pytest.raises(ValueError, match="Duplicate frame/track pair"):
        normalize_mot_rows(rows, 1, 100, 100)


def test_selects_highest_hota_then_identity_score():
    runs = [
        {
            "model": "baseline",
            "confidence": 0.3,
            "hota": 40,
            "idf1": 50,
            "mota": 60,
        },
        {
            "model": "baseline",
            "confidence": 0.4,
            "hota": 41,
            "idf1": 45,
            "mota": 70,
        },
        {
            "model": "candidate",
            "confidence": 0.3,
            "hota": 42,
            "idf1": 55,
            "mota": 65,
        },
    ]

    best = select_best_runs(runs)

    assert best["baseline"]["confidence"] == 0.4
    assert best["candidate"]["confidence"] == 0.3


def test_builds_locked_holdout_run_specs():
    mode, runs = build_run_specs(
        baseline_model="baseline.pt",
        candidate_model="candidate.pt",
        confidences=[0.1, 0.2],
        baseline_confidence=0.3,
        candidate_confidence=0.4,
    )

    assert mode == "locked_holdout"
    assert runs == [
        ("baseline", "baseline.pt", 0.3),
        ("mot17_a", "candidate.pt", 0.4),
    ]


def test_builds_run_specs_with_custom_candidate_label():
    _, runs = build_run_specs(
        baseline_model="baseline.pt",
        candidate_model="candidate.pt",
        confidences=[0.3],
        baseline_confidence=0.3,
        candidate_confidence=0.55,
        candidate_label="crowdhuman_a",
    )

    assert runs[1] == ("crowdhuman_a", "candidate.pt", 0.55)


@pytest.mark.parametrize("label", ["baseline", "../candidate", "candidate/test"])
def test_rejects_unsafe_or_reserved_candidate_label(label):
    with pytest.raises(ValueError, match="file-safe label"):
        build_run_specs(
            baseline_model="baseline.pt",
            candidate_model="candidate.pt",
            confidences=[0.3],
            baseline_confidence=None,
            candidate_confidence=None,
            candidate_label=label,
        )


def test_requires_both_locked_confidences():
    with pytest.raises(ValueError, match="must be used together"):
        build_run_specs(
            baseline_model="baseline.pt",
            candidate_model="candidate.pt",
            confidences=[0.3],
            baseline_confidence=0.3,
            candidate_confidence=None,
        )


def test_describes_benchmark_artifact_with_sha256(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"visionguard")

    description = describe_artifact(artifact)

    assert description == {
        "path": str(artifact),
        "size_bytes": 11,
        "sha256": hashlib.sha256(b"visionguard").hexdigest(),
    }


def test_reproducibility_metadata_allows_default_tracker_config(
    tmp_path,
    monkeypatch,
):
    artifacts = []
    for name in ("video.mp4", "gt.txt", "config.yaml", "model.pt"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        artifacts.append(path)
    monkeypatch.setattr(
        "scripts.evaluate_camera_tracking._git_metadata",
        lambda project_root: {"commit": "test", "dirty": False},
    )
    args = Namespace(
        video=str(artifacts[0]),
        ground_truth=str(artifacts[1]),
        config=str(artifacts[2]),
    )

    metadata = collect_reproducibility_metadata(
        args,
        tracker_config=None,
        run_specs=[("baseline", str(artifacts[3]), 0.3)],
    )

    assert metadata["artifacts"]["tracker_config"] is None


def test_resolves_ultralytics_default_tracker_config():
    tracker_config = resolve_tracker_config(None)

    assert tracker_config.name == "bytetrack.yaml"
    assert tracker_config.is_file()


def test_failed_tracking_run_preserves_previous_prediction(tmp_path, monkeypatch):
    prediction_path = tmp_path / "predictions.txt"
    prediction_path.write_text("previous\n", encoding="utf-8")

    class FakeCapture:
        def __init__(self):
            self.read_count = 0

        def isOpened(self):
            return True

        def read(self):
            self.read_count += 1
            return (True, object()) if self.read_count == 1 else (False, None)

        def release(self):
            pass

    class FailingTracker:
        def __init__(self, **kwargs):
            pass

        def track(self, frame):
            raise RuntimeError("tracking failed")

    monkeypatch.setattr(
        "scripts.evaluate_camera_tracking.cv2.VideoCapture",
        lambda path: FakeCapture(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_camera_tracking.YOLOByteTracker",
        FailingTracker,
    )

    with pytest.raises(RuntimeError, match="tracking failed"):
        run_tracking(
            video_path=tmp_path / "video.mp4",
            prediction_path=prediction_path,
            model_path="model.pt",
            confidence=0.3,
            target_classes=["person"],
            tracker_config=tmp_path / "tracker.yaml",
        )

    assert prediction_path.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".*.tmp")) == []
