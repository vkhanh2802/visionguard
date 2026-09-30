import pytest

from scripts.evaluate_camera_tracking import build_run_specs, select_best_runs
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


def test_requires_both_locked_confidences():
    with pytest.raises(ValueError, match="must be used together"):
        build_run_specs(
            baseline_model="baseline.pt",
            candidate_model="candidate.pt",
            confidences=[0.3],
            baseline_confidence=0.3,
            candidate_confidence=None,
        )
