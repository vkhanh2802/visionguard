from pathlib import Path

from src.database import SQLiteRepository
from src.event_jsonl import EventJsonlWriter
from src.events import Event
from src.pipeline import PipelineResult


class RunRecorder:
    def __init__(
        self,
        repository: SQLiteRepository,
        run_id: str,
        source_path: Path,
        output_path: Path,
        config_data: dict,
        event_writer: EventJsonlWriter | None = None,
        create_run: bool = True,
    ):
        self.repository = repository
        self.run_id = run_id
        self.event_writer = event_writer

        if create_run:
            self.repository.create_run(
                run_id=run_id,
                source_path=source_path,
                output_path=output_path,
                config_data=config_data,
            )

    def record_event(self, event: Event, frame_id: int) -> None:
        self.repository.record_event(
            run_id=self.run_id,
            event=event,
            frame_id=frame_id,
        )

        if self.event_writer is not None:
            self.event_writer.write_event(event, frame_id)

    def complete(self, result: PipelineResult, config_data: dict) -> None:
        self.repository.complete_run(self.run_id, result)

        if self.event_writer is not None:
            self.event_writer.write_metadata(config_data, result)

    def fail(self, error: Exception) -> None:
        self.repository.fail_run(self.run_id, str(error))

    def close(self) -> None:
        if self.event_writer is not None:
            self.event_writer.close()
