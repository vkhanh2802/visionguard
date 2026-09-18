from pathlib import Path

import numpy as np
import pytest
import sqlite3

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


def test_starts_queued_run(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    repository.create_run(
        "run-1",
        "input.mp4",
        "output.mp4",
        {},
        status="queued",
    )

    repository.start_run("run-1")

    assert repository.get_run("run-1")["status"] == "running"

    with pytest.raises(RuntimeError, match="Cannot start"):
        repository.start_run("run-1")


def test_migrates_existing_database_to_support_queued_runs(tmp_path: Path):
    database_path = tmp_path / "visionguard.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE analysis_runs (
            run_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            output_path TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('running', 'completed', 'failed')),
            created_at TEXT NOT NULL,
            completed_at TEXT,
            error_message TEXT,
            config_json TEXT NOT NULL,
            processed_frames INTEGER,
            source_fps REAL,
            effective_fps REAL,
            core_processing_fps REAL,
            end_to_end_fps REAL,
            elapsed_seconds REAL,
            stopped_early INTEGER,
            in_count INTEGER,
            out_count INTEGER,
            intrusion_count INTEGER,
            loitering_count INTEGER
        );
        CREATE TABLE events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            video_timestamp REAL NOT NULL,
            position_x INTEGER NOT NULL,
            position_y INTEGER NOT NULL,
            direction TEXT,
            zone_id TEXT,
            duration_seconds REAL,
            logged_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES analysis_runs(run_id) ON DELETE CASCADE
        );
        INSERT INTO analysis_runs (
            run_id, source_path, output_path, status, created_at, config_json
        ) VALUES ('legacy-run', 'input.mp4', 'output.mp4', 'completed', 'now', '{}');
        INSERT INTO events (
            run_id, frame_id, event_type, track_id, video_timestamp,
            position_x, position_y, logged_at
        ) VALUES ('legacy-run', 1, 'intrusion', 7, 1.0, 100, 200, 'now');
        """
    )
    connection.close()

    repository = SQLiteRepository(database_path)
    repository.create_run("queued-run", "input.mp4", "output.mp4", {}, status="queued")

    assert repository.get_run("legacy-run")["status"] == "completed"
    assert repository.list_events("legacy-run")[0]["track_id"] == 7
    assert repository.get_run("queued-run")["status"] == "queued"


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


def test_gets_run_analytics(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    repository.create_run("run-1", "input.mp4", "output.mp4", {})
    repository.complete_run("run-1", make_result())

    repository.record_event(
        "run-1",
        Event(
            event_type="intrusion",
            track_id=7,
            timestamp=5.0,
            position=(100, 200),
        ),
        frame_id=150,
    )
    repository.record_event(
        "run-1",
        Event(
            event_type="line_crossing",
            track_id=7,
            timestamp=7.0,
            position=(120, 220),
            direction="IN",
        ),
        frame_id=210,
    )

    analytics = repository.get_run_analytics("run-1")

    assert analytics["processed_frames"] == 100
    assert analytics["in_count"] == 2
    assert analytics["recorded_event_count"] == 2
    assert analytics["unique_track_count"] == 1
    assert analytics["first_event_timestamp"] == 5.0
    assert analytics["last_event_timestamp"] == 7.0
    assert analytics["event_counts"] == {
        "intrusion": 1,
        "line_crossing": 1,
    }


def test_marks_interrupted_runs_as_failed(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    repository.create_run("running-1", "input.mp4", "output.mp4", {})
    repository.create_run("completed-1", "input.mp4", "output.mp4", {})
    repository.complete_run("completed-1", make_result())

    recovered_count = repository.fail_interrupted_runs("API process restarted")

    running_run = repository.get_run("running-1")
    completed_run = repository.get_run("completed-1")
    assert recovered_count == 1
    assert running_run["status"] == "failed"
    assert running_run["error_message"] == "API process restarted"
    assert completed_run["status"] == "completed"
