import json
from datetime import datetime, timezone
from pathlib import Path

from src.events import Event
from src.pipeline.types import PipelineResult


class EventJsonlWriter:
    def __init__(self, event_path: Path, metadata_path: Path, run_id: str):
        self.event_path = Path(event_path)
        self.metadata_path = Path(metadata_path)
        self.run_id = run_id

        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.file = self.event_path.open(
            mode="w",
            encoding="utf-8",
        )

    def write_event(self, event: Event, frame_id: int) -> None:
        record = {
            "run_id": self.run_id,
            "frame_id": int(frame_id),
            "event_type": str(event.event_type),
            "track_id": int(event.track_id),
            "video_timestamp": float(event.timestamp),
            "position": {
                "x": int(event.position[0]),
                "y": int(event.position[1]),
            },
            "direction": event.direction,
            "zone_id": event.zone_id,
            "duration_seconds": (
                float(event.duration_seconds)
                if event.duration_seconds is not None
                else None
            ),
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def write_metadata(
        self,
        config_data: dict,
        result: PipelineResult,
    ) -> None:
        metadata = {
            "run_id": self.run_id,
            "status": "completed",
            "source_path": str(result.source_path),
            "output_path": str(result.output_path),
            "processed_frames": result.processed_frames,
            "source_fps": result.source_fps,
            "effective_fps": result.effective_fps,
            "core_processing_fps": result.core_processing_fps,
            "end_to_end_fps": result.end_to_end_fps,
            "elapsed_seconds": result.elapsed_seconds,
            "stopped_early": result.stopped_early,
            "counts": {
                "in": result.in_count,
                "out": result.out_count,
                "intrusion": result.intrusion_count,
                "loitering": result.loitering_count,
            },
            "config": config_data,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        self.metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        self.file.close()
