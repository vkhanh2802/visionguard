from pathlib import Path

from src.database import SQLiteRepository
from src.events import Event
from src.pipeline import PipelineResult
from src.run_recorder import RunRecorder


def make_result() -> PipelineResult:
    return PipelineResult(
        source_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        processed_frames=200,
        source_fps=30.0,
        effective_fps=30.0,
        core_processing_fps=20.0,
        end_to_end_fps=15.0,
        stopped_early=False,
        elapsed_seconds=10.0,
        in_count=0,
        out_count=0,
        intrusion_count=1,
        loitering_count=0,
    )


def test_recorder_persists_events_and_completes_run(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    recorder = RunRecorder(
        repository=repository,
        run_id="run-1",
        source_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        config_data={},
    )

    recorder.record_event(
        Event(
            event_type="intrusion",
            track_id=7,
            timestamp=5.0,
            position=(100, 200),
            zone_id="restricted-zone-1",
        ),
        frame_id=150,
    )
    recorder.complete(make_result(), config_data={})

    run = repository.get_run("run-1")
    events = repository.list_events("run-1")

    assert run["status"] == "completed"
    assert run["intrusion_count"] == 1
    assert len(events) == 1
    assert events[0]["frame_id"] == 150


def test_recorder_marks_failed_run(tmp_path: Path):
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    recorder = RunRecorder(
        repository=repository,
        run_id="run-1",
        source_path=Path("input.mp4"),
        output_path=Path("output.mp4"),
        config_data={},
    )

    recorder.fail(RuntimeError("Cannot open video"))

    run = repository.get_run("run-1")
    assert run["status"] == "failed"
    assert run["error_message"] == "Cannot open video"
