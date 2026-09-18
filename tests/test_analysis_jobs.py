from pathlib import Path

from src.analysis_jobs import execute_queued_analysis
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


def test_queued_job_starts_pipeline_and_completes_run(tmp_path: Path, monkeypatch):
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
                intrusion_count=1,
                loitering_count=0,
            )

    monkeypatch.setattr("src.analysis_service.VideoPipeline", FakeVideoPipeline)
    repository = SQLiteRepository(tmp_path / "visionguard.db")
    config = load_test_config()
    config_data = config.model_dump(mode="json")
    repository.create_run(
        "run-1",
        "input.mp4",
        "output.mp4",
        config_data,
        status="queued",
    )

    result = execute_queued_analysis(
        "run-1",
        config_data,
        "input.mp4",
        "output.mp4",
        str(repository.database_path),
    )

    assert result.processed_frames == 100
    assert repository.get_run("run-1")["status"] == "completed"
    assert repository.list_events("run-1")[0]["track_id"] == 7
