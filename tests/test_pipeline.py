from pathlib import Path

from src.config import AppConfig, load_config
from src.pipeline import VideoPipeline
from src.tracking import Track


def make_track(track_id: int, point: tuple[int, int]) -> Track:
    x, y = point

    return Track(
        track_id=track_id,
        bbox=(x - 10, y - 20, x + 10, y),
        confidence=0.9,
        class_id=0,
        class_name="person",
        centroid=(x, y - 10),
        bottom_center=point,
    )


def load_test_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "configs" / "default.yaml")

    data = config.model_dump(mode="python")
    data["output"]["display"] = False

    return AppConfig.model_validate(data)

def test_pipeline_processes_line_intrusion_and_loitering():
    pipeline = VideoPipeline(load_test_config())

    line_engine, intrusion_engine, loitering_engine = (
        pipeline._create_event_engines()
    )

    all_events = []

    frames = [
        (0, 0.0, (150, 350)),
        (1, 1.0, (150, 150)),
        (2, 2.0, (150, 150)),
        (3, 3.0, (150, 150)),
        (6, 6.0, (150, 150)),
    ]

    for frame_id, timestamp, point in frames:
        events = pipeline._process_events(
            tracks=[make_track(1, point)],
            frame_id=frame_id,
            timestamp=timestamp,
            line_engine=line_engine,
            intrusion_engine=intrusion_engine,
            loitering_engine=loitering_engine,
        )

        all_events.extend(events)

    assert [event.event_type for event in all_events] == [
        "intrusion",
        "line_crossing",
        "loitering",
    ]

    assert all_events[0].track_id == 1
    assert all_events[1].direction == "OUT"
    assert all_events[2].duration_seconds == 5.0

    assert line_engine.out_count == 1
    assert intrusion_engine.intrusion_count == 1
    assert loitering_engine.loitering_count == 1

def test_pipeline_does_not_create_disabled_engines():
    config = load_test_config()
    data = config.model_dump(mode="python")

    data["events"]["line_crossing"]["enabled"] = False
    data["events"]["intrusion"]["enabled"] = False
    data["events"]["loitering"]["enabled"] = False

    disabled_config = AppConfig.model_validate(data)
    pipeline = VideoPipeline(disabled_config)

    line_engine, intrusion_engine, loitering_engine = (
        pipeline._create_event_engines()
    )

    assert line_engine is None
    assert intrusion_engine is None
    assert loitering_engine is None

def test_new_engine_set_starts_with_clean_state():
    pipeline = VideoPipeline(load_test_config())

    first_line_engine, _, _ = pipeline._create_event_engines()
    first_line_engine.process(
        [make_track(1, (150, 350))],
        frame_id=0,
        timestamp=0.0,
    )

    second_line_engine, _, _ = pipeline._create_event_engines()

    assert second_line_engine.in_count == 0
    assert second_line_engine.out_count == 0
    assert second_line_engine.last_stable_side == {}