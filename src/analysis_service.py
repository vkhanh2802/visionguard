import logging
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from src.config import AppConfig
from src.database import SQLiteRepository
from src.event_jsonl import EventJsonlWriter
from src.events import Event
from src.pipeline import PipelineResult, VideoPipeline
from src.run_recorder import RunRecorder


@dataclass
class AnalysisJob:
    run_id: str
    config: AppConfig
    config_data: dict
    source_path: Path
    output_path: Path
    recorder: RunRecorder
    logger: logging.Logger


def create_analysis_job(
    config: AppConfig,
    source_path: str | Path,
    output_path: str | Path,
    repository: SQLiteRepository,
    logger: logging.Logger | None = None,
    run_id: str | None = None,
    create_run: bool = True,
) -> AnalysisJob:
    run_id = run_id or str(uuid4())
    source = Path(source_path)
    output = Path(output_path)
    config_data = config.model_dump(mode="json")
    event_writer = _create_event_writer(config, run_id)
    recorder = RunRecorder(
        repository=repository,
        run_id=run_id,
        source_path=source,
        output_path=output,
        config_data=config_data,
        event_writer=event_writer,
        create_run=create_run,
    )

    return AnalysisJob(
        run_id=run_id,
        config=config,
        config_data=config_data,
        source_path=source,
        output_path=output,
        recorder=recorder,
        logger=logger or logging.getLogger("visionguard"),
    )


def execute_analysis_job(job: AnalysisJob) -> PipelineResult:
    try:
        pipeline = VideoPipeline(
            config=job.config,
            event_handler=lambda event, frame_id: _record_event(
                event,
                frame_id,
                job.recorder,
                job.logger,
            ),
        )
        result = pipeline.run(
            source_path=job.source_path,
            output_path=job.output_path,
        )
        job.recorder.complete(result=result, config_data=job.config_data)
        return result
    except Exception as error:
        job.recorder.fail(error)
        raise
    finally:
        job.recorder.close()


def run_background_analysis(job: AnalysisJob) -> None:
    try:
        execute_analysis_job(job)
    except Exception:
        job.logger.exception("Pipeline failed for run_id=%s", job.run_id)


def _create_event_writer(
    config: AppConfig,
    run_id: str,
) -> EventJsonlWriter | None:
    if config.logging.event_jsonl_path is None:
        return None

    return EventJsonlWriter(
        event_path=config.logging.event_jsonl_path,
        metadata_path=config.logging.run_metadata_path,
        run_id=run_id,
    )


def _record_event(
    event: Event,
    frame_id: int,
    recorder: RunRecorder,
    logger: logging.Logger,
) -> None:
    recorder.record_event(event, frame_id)
    logger.info(
        "event=%s track_id=%s frame_id=%s video_timestamp=%.2f "
        "direction=%s zone_id=%s duration_seconds=%s",
        event.event_type,
        event.track_id,
        frame_id,
        event.timestamp,
        event.direction,
        event.zone_id,
        event.duration_seconds,
    )
