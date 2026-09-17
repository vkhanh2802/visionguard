import importlib
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.database import SQLiteRepository
from src.events import Event
from src.pipeline import PipelineResult


def create_client(tmp_path: Path) -> TestClient:
    database_path = tmp_path / "visionguard.db"
    repository = SQLiteRepository(database_path)
    repository.create_run("run-1", "input.mp4", "output.mp4", {})
    repository.record_event(
        "run-1",
        Event(
            event_type="intrusion",
            track_id=7,
            timestamp=5.0,
            position=(100, 200),
            zone_id="restricted-zone-1",
        ),
        frame_id=150,
    )

    return TestClient(create_app(database_path))


def test_health_returns_database_status(tmp_path: Path):
    response = create_client(tmp_path).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "connected"}


def test_root_lists_api_entrypoints(tmp_path: Path):
    response = create_client(tmp_path).get("/")

    assert response.status_code == 200
    assert response.json() == {
        "service": "VisionGuard API",
        "docs_url": "/docs",
        "health_url": "/health",
        "runs_url": "/runs",
        "events_url": "/events",
    }


def test_lists_runs(tmp_path: Path):
    response = create_client(tmp_path).get("/runs")

    assert response.status_code == 200
    assert response.json()["items"][0]["run_id"] == "run-1"


def test_gets_run_by_id(tmp_path: Path):
    response = create_client(tmp_path).get("/runs/run-1")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_returns_404_for_unknown_run(tmp_path: Path):
    response = create_client(tmp_path).get("/runs/missing")

    assert response.status_code == 404


def test_downloads_completed_run_output(tmp_path: Path):
    database_path = tmp_path / "visionguard.db"
    output_path = tmp_path / "annotated.mp4"
    output_content = b"mock mp4 content"
    output_path.write_bytes(output_content)

    repository = SQLiteRepository(database_path)
    repository.create_run("run-1", "input.mp4", output_path, {})
    repository.complete_run(
        "run-1",
        PipelineResult(
            source_path=Path("input.mp4"),
            output_path=output_path,
            processed_frames=100,
            source_fps=30.0,
            effective_fps=30.0,
            core_processing_fps=25.0,
            end_to_end_fps=20.0,
            elapsed_seconds=5.0,
            stopped_early=False,
            in_count=0,
            out_count=0,
            intrusion_count=0,
            loitering_count=0,
        ),
    )

    response = TestClient(create_app(database_path)).get("/runs/run-1/output")

    assert response.status_code == 200
    assert response.content == output_content
    assert "annotated.mp4" in response.headers["content-disposition"]


def test_rejects_output_for_unfinished_run(tmp_path: Path):
    database_path = tmp_path / "visionguard.db"
    repository = SQLiteRepository(database_path)
    repository.create_run("run-1", "input.mp4", "output.mp4", {})

    response = TestClient(create_app(database_path)).get("/runs/run-1/output")

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "Output is unavailable while run status is running."
    )


def test_gets_run_analytics(tmp_path: Path):
    response = create_client(tmp_path).get("/runs/run-1/analytics")

    assert response.status_code == 200
    assert response.json()["run_id"] == "run-1"
    assert response.json()["events"]["recorded_event_count"] == 1
    assert response.json()["events"]["unique_track_count"] == 1
    assert response.json()["events"]["by_type"] == {"intrusion": 1}


def test_returns_404_for_missing_run_analytics(tmp_path: Path):
    response = create_client(tmp_path).get("/runs/missing/analytics")

    assert response.status_code == 404


def test_filters_events(tmp_path: Path):
    response = create_client(tmp_path).get(
        "/events",
        params={"run_id": "run-1", "event_type": "intrusion"},
    )

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["track_id"] == 7


def test_rejects_invalid_pagination(tmp_path: Path):
    response = create_client(tmp_path).get("/events", params={"limit": 0})

    assert response.status_code == 422


def test_accepts_analysis_job(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "visionguard.db"
    captured = {}
    api_module = importlib.import_module("src.api.app")

    def fake_create_analysis_job(**kwargs):
        captured["config"] = kwargs["config"]
        captured["source_path"] = kwargs["source_path"]
        captured["output_path"] = kwargs["output_path"]
        return SimpleNamespace(run_id="run-analysis")

    def fake_run_background_analysis(job):
        captured["background_run_id"] = job.run_id

    monkeypatch.setattr(api_module, "create_analysis_job", fake_create_analysis_job)
    monkeypatch.setattr(api_module, "run_background_analysis", fake_run_background_analysis)
    client = TestClient(create_app(database_path))

    response = client.post(
        "/analyze",
        json={
            "source_path": "data/videos/test.mp4",
            "output_path": "data/outputs/api-output.mp4",
        },
    )

    assert response.status_code == 202
    assert response.json() == {"run_id": "run-analysis", "status": "running"}
    assert captured["source_path"] == "data/videos/test.mp4"
    assert captured["output_path"] == "data/outputs/api-output.mp4"
    assert not captured["config"].output.display
    assert captured["background_run_id"] == "run-analysis"


def test_rejects_missing_analysis_config(tmp_path: Path):
    response = TestClient(create_app(tmp_path / "visionguard.db")).post(
        "/analyze",
        json={
            "source_path": "input.mp4",
            "output_path": "output.mp4",
            "config_path": "configs/missing.yaml",
        },
    )

    assert response.status_code == 422


def test_startup_marks_interrupted_runs_as_failed(tmp_path: Path):
    database_path = tmp_path / "visionguard.db"
    repository = SQLiteRepository(database_path)
    repository.create_run("running-1", "input.mp4", "output.mp4", {})

    with TestClient(create_app(database_path)) as client:
        response = client.get("/runs/running-1")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert response.json()["error_message"] == (
        "Analysis interrupted because the API process restarted."
    )
