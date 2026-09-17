from pathlib import Path

import pytest

from src.analysis_service import create_analysis_job, execute_analysis_job
from src.config import AppConfig, load_config
from src.database import SQLiteRepository
from src.events import Event
from src.pipeline import PipelineResult


def load_test_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")
    config_data = config.model_dump(mode="python")
    config_data["output"]["display"] = False
    return AppConfig.model_validate(config_data)


def make_result(source_path: Path, output_path: Path) -> PipelineResult:
    return PipelineResult(
        source_path=source_path,
        output_path=output_path,
        processed_frames=100,
        source_fps=30.0,
        effective_fps=30.0,
        core_processing_fps=25.0,
        end_to_end_fps=20.0,
        elapsed_seconds=5.0,
        stopped_early=False,
        in_count=2,
        out_count=1,
        intrusion_count=3,
        loitering_count=0,
    )


def test_executes_pipeline_and_completes_run(tmp_path: Path, monkeypatch):
    class FakeVideoPipeline:
        def __init__(self, config: AppConfig, event_handler):
            self.event_handler = event_handler

        def run(self, source_path: Path, output_path: Path) -> PipelineResult:
            self.event_handler(
                Event(
                    event_type="intrusion",
                    track_id=7,
                    timestamp=5.0,
                    position=(100, 200),
                    zone_id="restricted-zone-1",
                ),
                150,
            )
            return make_result(source_path, output_path)

    monkeypatch.setattr("src.analysis_service.VideoPipeline", FakeVideoPipeline)
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    job = create_analysis_job(
        config=load_test_config(),
        source_path="input.mp4",
        output_path="output.mp4",
        repository=repository,
    )

    result = execute_analysis_job(job)

    assert result.processed_frames == 100
    assert repository.get_run(job.run_id)["status"] == "completed"
    assert repository.list_events(job.run_id)[0]["track_id"] == 7


def test_marks_run_failed_when_pipeline_raises(tmp_path: Path, monkeypatch):
    class FailingVideoPipeline:
        def __init__(self, config: AppConfig, event_handler):
            pass

        def run(self, source_path: Path, output_path: Path) -> PipelineResult:
            raise RuntimeError("Cannot open video")

    monkeypatch.setattr("src.analysis_service.VideoPipeline", FailingVideoPipeline)
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    job = create_analysis_job(
        config=load_test_config(),
        source_path="missing.mp4",
        output_path="output.mp4",
        repository=repository,
    )

    with pytest.raises(RuntimeError, match="Cannot open video"):
        execute_analysis_job(job)

    run = repository.get_run(job.run_id)
    assert run["status"] == "failed"
    assert run["error_message"] == "Cannot open video"
