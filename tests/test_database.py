from pathlib import Path

import numpy as np
import pytest

from src.database import SQLiteRepository
from src.events import Event
from src.pipeline import PipelineResult


def make_result() -> PipelineResult:
    return PipelineResult(
        source_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        processed_frames=100,
        source_fps=30.0,
        effective_fps=30.0,
        core_processing_fps=25.0,
        end_to_end_fps=20.0,
        stopped_early=False,
        elapsed_seconds=5.0,
        in_count=2,
        out_count=1,
        intrusion_count=3,
        loitering_count=1,
    )


def test_creates_and_completes_run(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")

    repository.create_run(
        run_id="run-1",
        source_path="input.mp4",
        output_path="output.mp4",
        config_data={"detection": {"confidence": 0.4}},
    )

    assert repository.get_run("run-1")["status"] == "running"

    repository.complete_run("run-1", make_result())

    run = repository.get_run("run-1")
    assert run["status"] == "completed"
    assert run["processed_frames"] == 100
    assert run["intrusion_count"] == 3
    assert run["stopped_early"] == 0


def test_records_and_filters_events(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    repository.create_run("run-1", "input.mp4", "output.mp4", {})

    repository.record_event(
        "run-1",
        Event(
            event_type="intrusion",
            track_id=7,
            timestamp=5.0,
            position=(np.int64(120), np.int64(240)),
            zone_id="restricted-zone-1",
        ),
        frame_id=150,
    )

    repository.record_event(
        "run-1",
        Event(
            event_type="line_crossing",
            track_id=8,
            timestamp=7.0,
            position=(150, 300),
            direction="IN",
        ),
        frame_id=210,
    )

    events = repository.list_events("run-1")
    intrusion_events = repository.list_events(
        "run-1",
        event_type="intrusion",
    )

    assert len(events) == 2
    assert events[0]["position_x"] == 120
    assert events[0]["position_y"] == 240
    assert events[1]["direction"] == "IN"
    assert len(intrusion_events) == 1
    assert intrusion_events[0]["track_id"] == 7


def test_unknown_run_cannot_be_completed(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")

    with pytest.raises(KeyError, match="Unknown run_id"):
        repository.complete_run("missing", make_result())


def test_marks_run_as_failed(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    repository.create_run("run-1", "input.mp4", "output.mp4", {})

    repository.fail_run("run-1", "Cannot open video")

    run = repository.get_run("run-1")
    assert run["status"] == "failed"
    assert run["error_message"] == "Cannot open video"


def test_event_pagination_arguments_are_validated(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")

    with pytest.raises(ValueError, match="at least 1"):
        repository.list_events("run-1", limit=0)

    with pytest.raises(ValueError, match="non-negative"):
        repository.list_events("run-1", offset=-1)
