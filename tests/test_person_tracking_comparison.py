from pathlib import Path

import pytest

from scripts.compare_person_tracking_videos import (
    build_tracking_only_config,
    discover_videos,
    select_models,
    summarize_result,
)
from src.config import load_config
from src.pipeline import PipelineResult


def test_discovers_selected_videos(tmp_path: Path):
    (tmp_path / "1.mp4").touch()
    (tmp_path / "2.MKV").touch()
    (tmp_path / "notes.txt").touch()

    videos = discover_videos(tmp_path, ["2.MKV"])

    assert videos == [tmp_path / "2.MKV"]


def test_builds_tracking_only_config():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")

    overridden = build_tracking_only_config(config, "candidate.pt", 0.25)

    assert overridden.detection.model_path == "candidate.pt"
    assert overridden.detection.confidence == 0.25
    assert overridden.tracking.continuity_enabled is False
    assert overridden.tracking.duplicate_suppression_enabled is False
    assert overridden.events.line_crossing.enabled is False
    assert overridden.events.intrusion.enabled is False
    assert overridden.events.loitering.enabled is False
    assert overridden.events.zones == {}
    assert overridden.output.display is False


def test_selects_only_baseline_for_final_regression():
    models = select_models("baseline.pt", "candidate.pt", baseline_only=True)

    assert models == (("baseline", "baseline.pt"),)


def test_final_person_tracking_config_is_isolated():
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "person_tracking_final.yaml")

    assert config.detection.model_path == "yolo26n.pt"
    assert config.detection.confidence == 0.3
    assert config.tracking.tracker_config == Path("configs/bytetrack_week6.yaml")
    assert config.tracking.continuity_enabled is False
    assert config.tracking.duplicate_suppression_enabled is False
    assert config.events.line_crossing.enabled is False
    assert config.events.intrusion.enabled is False
    assert config.events.loitering.enabled is False
    assert config.events.zones == {}
    assert config.output.display is False
    assert config.output.async_writer is True
    assert config.output.writer_queue_size == 4
    assert config.output.encoder == "ffmpeg_nvenc"
    assert config.output.nvenc_quality == 23


def test_summarizes_tracking_proxies():
    result = PipelineResult(
        source_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        processed_frames=100,
        source_fps=25.0,
        effective_fps=25.0,
        core_processing_fps=30.0,
        end_to_end_fps=20.0,
        stopped_early=False,
        elapsed_seconds=5.0,
        in_count=0,
        out_count=0,
        intrusion_count=0,
        loitering_count=0,
        tracking_diagnostics={
            "total_track_count": 2,
            "tracks_with_gaps": 1,
            "total_missing_frames": 4,
            "max_gap_frames": 3,
            "median_observed_frames": 25.0,
            "track_lifetimes": [
                {"observed_frames": 40},
                {"observed_frames": 10},
            ],
            "timing": {"tracking_fps": 35.0},
        },
    )

    summary = summarize_result(result, "input.mp4", "candidate", "candidate.pt")

    assert summary["observed_track_frames"] == 50
    assert summary["average_active_tracks"] == 0.5
    assert summary["track_churn_per_1000_observations"] == 40.0
    assert summary["tracks_with_gaps_percent"] == 50.0
    assert summary["end_to_end_fps"] == 20.0
    assert summary["core_processing_fps"] == 30.0
    assert summary["tracking_fps"] == 35.0
