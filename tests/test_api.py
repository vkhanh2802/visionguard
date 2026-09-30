from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
from redis.exceptions import RedisError

from src.api.app import _ffmpeg_capabilities, create_app
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
        "readiness_url": "/readiness",
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
    assert response.json()["tracking"] is None


def test_returns_404_for_missing_run_analytics(tmp_path: Path):
    response = create_client(tmp_path).get("/runs/missing/analytics")

    assert response.status_code == 404


def test_returns_persisted_tracking_diagnostics(tmp_path: Path):
    database_path = tmp_path / "visionguard.db"
    repository = SQLiteRepository(database_path)
    repository.create_run("run-1", "input.mp4", "output.mp4", {})
    repository.complete_run(
        "run-1",
        PipelineResult(
            source_path=Path("input.mp4"),
            output_path=Path("output.mp4"),
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
            tracking_diagnostics={
                "total_track_count": 2,
                "new_track_count_in_roi": 1,
                "tracks_with_gaps": 1,
                "total_missing_frames": 3,
                "max_gap_frames": 2,
                "median_observed_frames": 10.0,
                "track_lifetimes": [],
                "timing": {
                    "read_seconds": 1.0,
                    "tracking_seconds": 2.0,
                    "analytics_seconds": 0.5,
                    "drawing_seconds": 0.25,
                    "write_enqueue_seconds": 0.1,
                    "encoding_seconds": 1.5,
                    "writer_flush_seconds": 0.2,
                    "frame_loop_seconds": 4.0,
                    "tracking_fps": 50.0,
                },
            },
        ),
    )

    response = TestClient(create_app(database_path)).get("/runs/run-1/analytics")

    assert response.status_code == 200
    assert response.json()["tracking"]["new_track_count_in_roi"] == 1
    assert response.json()["tracking"]["timing"]["tracking_fps"] == 50.0


def test_readiness_checks_job_processing_dependencies(tmp_path: Path, monkeypatch):
    class ReadyConnection:
        def ping(self):
            return True

    class ReadyQueue:
        connection = ReadyConnection()

    settings = ApiSettings(
        source_root=tmp_path,
        output_root=tmp_path / "outputs",
        config_root=tmp_path,
    )
    class ReadyWorker:
        last_heartbeat = datetime.now(timezone.utc)

        def get_state(self):
            return "idle"

    monkeypatch.setattr("src.api.app.Worker.all", lambda **kwargs: [ReadyWorker()])
    monkeypatch.setattr("src.api.app._ffmpeg_capabilities", lambda: (True, True))
    response = TestClient(
        create_app(tmp_path / "visionguard.db", settings, ReadyQueue())
    ).get("/readiness")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "database_writable": True,
        "redis_connected": True,
        "worker_available": True,
        "output_writable": True,
        "ffmpeg_available": True,
        "nvenc_available": True,
    }


def test_readiness_reports_optional_encoder_capabilities(tmp_path: Path, monkeypatch):
    class ReadyConnection:
        def ping(self):
            return True

    class ReadyQueue:
        connection = ReadyConnection()

    class ReadyWorker:
        last_heartbeat = datetime.now(timezone.utc)

        def get_state(self):
            return "busy"

    settings = ApiSettings(
        source_root=tmp_path,
        output_root=tmp_path / "outputs",
        config_root=tmp_path,
    )
    monkeypatch.setattr("src.api.app.Worker.all", lambda **kwargs: [ReadyWorker()])
    monkeypatch.setattr("src.api.app._ffmpeg_capabilities", lambda: (False, False))

    response = TestClient(
        create_app(tmp_path / "visionguard.db", settings, ReadyQueue())
    ).get("/readiness")

    assert response.status_code == 200
    assert response.json()["ffmpeg_available"] is False
    assert response.json()["nvenc_available"] is False


def test_readiness_rejects_worker_without_live_state(tmp_path: Path, monkeypatch):
    class ReadyConnection:
        def ping(self):
            return True

    class ReadyQueue:
        connection = ReadyConnection()

    class StaleWorker:
        last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=5)
        job_monitoring_interval = 30

        def get_state(self):
            return "busy"

    settings = ApiSettings(
        source_root=tmp_path,
        output_root=tmp_path / "outputs",
        config_root=tmp_path,
    )
    monkeypatch.setattr("src.api.app.Worker.all", lambda **kwargs: [StaleWorker()])
    monkeypatch.setattr("src.api.app._ffmpeg_capabilities", lambda: (True, True))

    response = TestClient(
        create_app(tmp_path / "visionguard.db", settings, ReadyQueue())
    ).get("/readiness")

    assert response.status_code == 503
    assert response.json()["detail"]["worker_available"] is False


def test_readiness_returns_503_when_redis_is_unavailable(tmp_path: Path, monkeypatch):
    class FailedConnection:
        def ping(self):
            raise RedisError("unavailable")

    class FailedQueue:
        connection = FailedConnection()

    settings = ApiSettings(
        source_root=tmp_path,
        output_root=tmp_path / "outputs",
        config_root=tmp_path,
    )
    monkeypatch.setattr("src.api.app._ffmpeg_capabilities", lambda: (True, True))
    response = TestClient(
        create_app(tmp_path / "visionguard.db", settings, FailedQueue())
    ).get("/readiness")

    assert response.status_code == 503
    assert response.json()["detail"]["redis_connected"] is False
    assert response.json()["detail"]["worker_available"] is False


def test_ffmpeg_capabilities_probe_nvenc_runtime(monkeypatch):
    class EncoderResult:
        stdout = " V....D h264_nvenc NVIDIA NVENC H.264 encoder"

    class FailedProbeResult:
        returncode = 1

    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return EncoderResult() if len(calls) == 1 else FailedProbeResult()

    monkeypatch.setattr("src.api.app.shutil.which", lambda executable: "ffmpeg")
    monkeypatch.setattr("src.api.app.subprocess.run", fake_run)
    _ffmpeg_capabilities.cache_clear()
    try:
        assert _ffmpeg_capabilities() == (True, False)
        assert len(calls) == 2
        assert "color=size=320x180:rate=1" in calls[1]
    finally:
        _ffmpeg_capabilities.cache_clear()


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
    source_root = tmp_path / "videos"
    output_root = tmp_path / "outputs"
    config_root = Path(__file__).resolve().parents[1] / "configs"
    source_root.mkdir()
    output_root.mkdir()
    source_path = source_root / "test.mp4"
    source_path.write_bytes(b"")
    captured = {}

    class RecordingQueue:
        def enqueue(self, function, *args, **kwargs):
            captured["function"] = function
            captured["args"] = args
            captured["kwargs"] = kwargs

    settings = ApiSettings(
        source_root=source_root,
        output_root=output_root,
        config_root=config_root,
    )
    client = TestClient(create_app(database_path, settings, RecordingQueue()))

    response = client.post(
        "/analyze",
        json={
            "source_path": str(source_path),
            "config_path": str(config_root / "default.yaml"),
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert captured["args"][0] == body["run_id"]
    assert captured["args"][2] == str(source_path.resolve())
    assert captured["args"][3] == str(
        (output_root / body["run_id"] / "annotated.mp4").resolve()
    )
    assert captured["kwargs"]["job_id"] == body["run_id"]
    run = SQLiteRepository(database_path).get_run(body["run_id"])
    assert run["status"] == "queued"
    assert run["output_path"] == captured["args"][3]


def test_scopes_analysis_artifacts_to_run_id(tmp_path: Path):
    source_root = tmp_path / "videos"
    output_root = tmp_path / "outputs"
    config_root = tmp_path / "configs"
    source_root.mkdir()
    output_root.mkdir()
    config_root.mkdir()
    (source_root / "test.mp4").write_bytes(b"")
    config_path = config_root / "config.yaml"
    config_path.write_text(
        """\
detection:
  model_path: yolo26n.pt
  confidence: 0.4
  target_classes: [person]
tracking:
  history_length: 30
  max_missing_frames: 30
events:
  line_crossing:
    enabled: true
    start: [50, 300]
    end: [600, 300]
  zones:
    restricted-zone-1:
      polygon: [[100, 100], [200, 100], [200, 200], [100, 200]]
  intrusion:
    enabled: true
    zone_id: restricted-zone-1
  loitering:
    enabled: true
    zone_id: restricted-zone-1
    dwell_threshold_seconds: 5.0
output:
  display: false
  codec: mp4v
logging:
  level: INFO
  event_jsonl_path: PLACEHOLDER/events.jsonl
  run_metadata_path: PLACEHOLDER/metadata.json
""".replace("PLACEHOLDER", output_root.as_posix()),
        encoding="utf-8",
    )
    captured = {}

    class RecordingQueue:
        def enqueue(self, function, *args, **kwargs):
            captured.setdefault("calls", []).append(args)

    settings = ApiSettings(
        source_root=source_root,
        output_root=output_root,
        config_root=config_root,
    )
    response = TestClient(
        create_app(tmp_path / "visionguard.db", settings, RecordingQueue())
    ).post(
        "/analyze",
        json={
            "source_path": str(source_root / "test.mp4"),
            "config_path": str(config_path),
        },
    )

    run_id = response.json()["run_id"]
    first_call = captured["calls"][0]
    logging_config = first_call[1]["logging"]
    assert response.status_code == 202
    assert Path(first_call[3]) == output_root / run_id / "annotated.mp4"
    assert Path(logging_config["event_jsonl_path"]) == output_root / run_id / "events.jsonl"
    assert Path(logging_config["run_metadata_path"]) == output_root / run_id / "metadata.json"

    second_response = TestClient(
        create_app(tmp_path / "visionguard.db", settings, RecordingQueue())
    ).post(
        "/analyze",
        json={
            "source_path": str(source_root / "test.mp4"),
            "config_path": str(config_path),
        },
    )
    second_run_id = second_response.json()["run_id"]
    second_call = captured["calls"][1]
    assert Path(second_call[3]) == output_root / second_run_id / "annotated.mp4"
    assert first_call[3] != second_call[3]


def test_rejects_colliding_analysis_artifact_names(tmp_path: Path):
    source_root = tmp_path / "videos"
    output_root = tmp_path / "outputs"
    config_root = tmp_path / "configs"
    source_root.mkdir()
    output_root.mkdir()
    config_root.mkdir()
    source_path = source_root / "test.mp4"
    source_path.write_bytes(b"")
    config_path = config_root / "config.yaml"
    default_config = (
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    ).read_text(encoding="utf-8")
    config_path.write_text(
        default_config.replace(
            "event_jsonl_path: null\n  run_metadata_path: null",
            "event_jsonl_path: artifacts/events.jsonl\n"
            "  run_metadata_path: artifacts/EVENTS.JSONL",
        ),
        encoding="utf-8",
    )
    settings = ApiSettings(
        source_root=source_root,
        output_root=output_root,
        config_root=config_root,
    )

    response = TestClient(create_app(tmp_path / "visionguard.db", settings)).post(
        "/analyze",
        json={"source_path": str(source_path), "config_path": str(config_path)},
    )

    assert response.status_code == 422
    assert "names must be unique" in response.json()["detail"]


def test_rejects_client_supplied_analysis_output_path(tmp_path: Path):
    response = TestClient(create_app(tmp_path / "visionguard.db")).post(
        "/analyze",
        json={
            "source_path": "input.mp4",
            "config_path": "configs/default.yaml",
            "output_path": "shared.mp4",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_rejects_missing_analysis_config(tmp_path: Path):
    response = TestClient(create_app(tmp_path / "visionguard.db")).post(
        "/analyze",
        json={
            "source_path": "input.mp4",
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
