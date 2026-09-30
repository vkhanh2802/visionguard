from pathlib import Path

import pytest

from scripts.evaluate_mot17 import apply_benchmark_overrides
from src.config import AppConfig, load_config
from src.evaluation import (
    combine_evaluations,
    evaluate_mot_files,
    format_mot_prediction,
    load_sequence_info,
)
from src.tracking import Track


def write_mot_file(path: Path, rows: list[str]) -> Path:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_loads_mot_sequence_info(tmp_path: Path):
    path = tmp_path / "seqinfo.ini"
    path.write_text(
        """[Sequence]
name=MOT17-02-FRCNN
imDir=img1
frameRate=30
seqLength=600
imWidth=1920
imHeight=1080
imExt=.jpg
""",
        encoding="utf-8",
    )

    info = load_sequence_info(path)

    assert info.name == "MOT17-02-FRCNN"
    assert info.frame_rate == 30
    assert info.length == 600
    assert info.image_extension == ".jpg"


def test_formats_track_as_mot_prediction():
    track = Track(
        track_id=7,
        bbox=(10, 20, 40, 80),
        confidence=0.875,
        class_id=0,
        class_name="person",
        centroid=(25, 50),
        bottom_center=(25, 80),
    )

    row = format_mot_prediction(3, track)

    assert row == "3,7,10.00,20.00,30.00,60.00,0.875000,-1,-1,-1\n"


def test_perfect_tracking_scores_full_metrics(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        [
            "1,1,0,0,10,10,1,1,1",
            "2,1,1,0,10,10,1,1,1",
        ],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        [
            "1,5,0,0,10,10,0.9,-1,-1,-1",
            "2,5,1,0,10,10,0.9,-1,-1,-1",
        ],
    )

    evaluation = evaluate_mot_files(ground_truth, predictions)
    metrics = evaluation.metrics()

    assert metrics["hota"] == 100.0
    assert metrics["mota"] == 100.0
    assert metrics["idf1"] == 100.0
    assert metrics["id_switches"] == 0


def test_counts_identity_switch(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        [
            "1,1,0,0,10,10,1,1,1",
            "2,1,0,0,10,10,1,1,1",
        ],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        [
            "1,5,0,0,10,10,0.9,-1,-1,-1",
            "2,6,0,0,10,10,0.9,-1,-1,-1",
        ],
    )

    evaluation = evaluate_mot_files(ground_truth, predictions)
    metrics = evaluation.metrics()

    assert metrics["id_switches"] == 1
    assert metrics["mota"] == 50.0
    assert metrics["idf1"] == 50.0


def test_identity_matching_uses_all_eligible_pairs_across_sequence(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        [
            "1,1,0,0,10,10,1,1,1",
            "1,2,2,0,10,10,1,1,1",
            "2,1,0,0,10,10,1,1,1",
            "2,2,2,0,10,10,1,1,1",
        ],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        [
            "1,10,0,0,10,10,0.9,-1,-1,-1",
            "1,20,2,0,10,10,0.9,-1,-1,-1",
            "2,10,2,0,10,10,0.9,-1,-1,-1",
            "2,20,0,0,10,10,0.9,-1,-1,-1",
        ],
    )

    metrics = evaluate_mot_files(ground_truth, predictions).metrics()

    assert metrics["idf1"] == 100.0


def test_identity_switch_after_unmatched_frame_uses_previous_timestep(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        [
            "1,1,0,0,10,10,1,1,1",
            "2,1,0,0,10,10,1,1,1",
            "3,1,0,0,10,10,1,1,1",
        ],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        [
            "1,10,0,0,10,10,0.9,-1,-1,-1",
            "2,99,100,0,10,10,0.9,-1,-1,-1",
            "3,10,2,0,10,10,0.9,-1,-1,-1",
            "3,20,0,0,10,10,0.9,-1,-1,-1",
        ],
    )

    metrics = evaluate_mot_files(ground_truth, predictions).metrics()

    assert metrics["id_switches"] == 1
    assert metrics["fragmentations"] == 1


def test_fragmentation_counts_track_reappearing_after_other_tracks(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        [
            "1,1,0,0,10,10,1,1,1",
            "2,2,20,0,10,10,1,1,1",
            "3,1,0,0,10,10,1,1,1",
        ],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        [
            "1,10,0,0,10,10,0.9,-1,-1,-1",
            "2,20,20,0,10,10,0.9,-1,-1,-1",
            "3,10,0,0,10,10,0.9,-1,-1,-1",
        ],
    )

    metrics = evaluate_mot_files(ground_truth, predictions).metrics()

    assert metrics["fragmentations"] == 1


def test_ignores_prediction_on_distractor(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        ["1,1,0,0,10,10,0,7,1"],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        ["1,5,0,0,10,10,0.9,-1,-1,-1"],
    )

    evaluation = evaluate_mot_files(ground_truth, predictions)
    metrics = evaluation.metrics()

    assert metrics["ground_truth_detections"] == 0
    assert metrics["predicted_detections"] == 0
    assert metrics["false_positives"] == 0


def test_distractor_preprocessing_matches_trackeval_tie_breaking():
    fixture_dir = Path(__file__).parent / "fixtures" / "mot_assignment_tie"

    metrics = evaluate_mot_files(
        fixture_dir / "gt.txt",
        fixture_dir / "predictions.txt",
    ).metrics()

    assert metrics["predicted_detections"] == 2
    assert metrics["id_switches"] == 0
    assert metrics["idf1"] == 100.0


def test_does_not_ignore_small_prediction_contained_by_large_distractor(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        ["1,1,0,0,100,100,1,7,1"],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        ["1,5,0,0,10,10,0.9,-1,-1,-1"],
    )

    metrics = evaluate_mot_files(ground_truth, predictions).metrics()

    assert metrics["predicted_detections"] == 1
    assert metrics["false_positives"] == 1


def test_hota_localization_is_full_when_there_are_no_matches(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        ["1,1,0,0,10,10,1,1,1"],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        ["1,5,100,100,10,10,0.9,-1,-1,-1"],
    )

    metrics = evaluate_mot_files(ground_truth, predictions).metrics()

    assert metrics["hota"] == 0.0
    assert metrics["loca"] == 100.0


def test_combines_sequence_counts(tmp_path: Path):
    ground_truth = write_mot_file(
        tmp_path / "gt.txt",
        ["1,1,0,0,10,10,1,1,1"],
    )
    predictions = write_mot_file(
        tmp_path / "predictions.txt",
        ["1,5,0,0,10,10,0.9,-1,-1,-1"],
    )
    evaluation = evaluate_mot_files(ground_truth, predictions)

    combined = combine_evaluations([evaluation, evaluation]).metrics()

    assert combined["ground_truth_detections"] == 2
    assert combined["true_positives"] == 2
    assert combined["hota"] == pytest.approx(100.0)


def test_applies_mot17_benchmark_overrides():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "week6.yaml")
    config_data = config.model_dump(mode="python")
    config_data["tracking"]["continuity_enabled"] = True
    config = AppConfig.model_validate(config_data)

    overridden = apply_benchmark_overrides(
        config,
        confidence=0.1,
        disable_continuity=True,
        model_path="trained-person-detector.pt",
    )

    assert overridden.detection.model_path == "trained-person-detector.pt"
    assert overridden.detection.confidence == 0.1
    assert overridden.tracking.continuity_enabled is False
    assert config.detection.model_path != "trained-person-detector.pt"
    assert config.detection.confidence == 0.4
    assert config.tracking.continuity_enabled is True
