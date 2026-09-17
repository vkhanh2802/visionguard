import importlib
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.database import SQLiteRepository
from src.events import Event


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
