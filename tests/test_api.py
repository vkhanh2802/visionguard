from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.settings import ApiSettings
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

    settings = ApiSettings(
        source_root=tmp_path,
        output_root=tmp_path,
        config_root=Path("configs"),
    )
    response = TestClient(create_app(database_path, settings)).get(
        "/runs/run-1/output"
    )

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

    class RecordingQueue:
        def enqueue(self, function, *args, **kwargs):
            captured["function"] = function
            captured["args"] = args
            captured["kwargs"] = kwargs

    client = TestClient(create_app(database_path, queue=RecordingQueue()))

    response = client.post(
        "/analyze",
        json={
            "source_path": "C:/VisionGuard/videos/test.mp4",
            "output_path": "C:/VisionGuard/outputs/api-output.mp4",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert captured["args"][0] == body["run_id"]
    assert captured["args"][2] == str(Path("C:/VisionGuard/videos/test.mp4").resolve())
    assert captured["args"][3] == str(
        Path("C:/VisionGuard/outputs/api-output.mp4").resolve()
    )
    assert captured["kwargs"]["job_id"] == body["run_id"]
    assert SQLiteRepository(database_path).get_run(body["run_id"])["status"] == "queued"


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


def test_rejects_analysis_paths_outside_configured_roots(tmp_path: Path):
    source_root = tmp_path / "videos"
    output_root = tmp_path / "outputs"
    source_root.mkdir()
    output_root.mkdir()
    settings = ApiSettings(
        source_root=source_root,
        output_root=output_root,
        config_root=Path("configs"),
    )
    client = TestClient(create_app(tmp_path / "visionguard.db", settings))

    response = client.post(
        "/analyze",
        json={
            "source_path": str(tmp_path / "outside.mp4"),
            "output_path": str(output_root / "output.mp4"),
        },
    )

    assert response.status_code == 422
    assert "Source video path must be inside" in response.json()["detail"]


def test_allows_dashboard_cors_origin(tmp_path: Path):
    response = TestClient(create_app(tmp_path / "visionguard.db")).options(
        "/runs",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_loads_api_settings_from_environment(monkeypatch):
    monkeypatch.setenv("VISIONGUARD_SOURCE_ROOT", "custom/videos")
    monkeypatch.setenv("VISIONGUARD_OUTPUT_ROOT", "custom/outputs")
    monkeypatch.setenv("VISIONGUARD_CONFIG_ROOT", "custom/configs")
    monkeypatch.setenv(
        "VISIONGUARD_ALLOWED_ORIGINS",
        "https://dashboard.example, http://localhost:5173",
    )

    settings = ApiSettings.from_environment()

    assert settings.source_root == Path("custom/videos")
    assert settings.output_root == Path("custom/outputs")
    assert settings.config_root == Path("custom/configs")
    assert settings.allowed_origins == (
        "https://dashboard.example",
        "http://localhost:5173",
    )


def test_api_start_does_not_fail_worker_run(tmp_path: Path):
    database_path = tmp_path / "visionguard.db"
    repository = SQLiteRepository(database_path)
    repository.create_run("running-1", "input.mp4", "output.mp4", {})

    with TestClient(create_app(database_path)) as client:
        response = client.get("/runs/running-1")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
