import argparse
from pathlib import Path
from uuid import uuid4

import logging 
from src.database import SQLiteRepository
from src.event_jsonl import EventJsonlWriter
from src.logging_config import configure_logging

from src.config import AppConfig, load_config
from src.events import Event
from src.pipeline import PipelineResult, VideoPipeline
from src.run_recorder import RunRecorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionGuard video analytics pipeline",
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/output.mp4",
        help="Path to annotated output video",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override detection.model_path from config",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detection.confidence from config",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable preview window",
    )

    return parser.parse_args()


def apply_cli_overrides(
    config: AppConfig,
    args: argparse.Namespace,
) -> AppConfig:
    data = config.model_dump(mode="python")

    if args.model is not None:
        data["detection"]["model_path"] = args.model

    if args.conf is not None:
        data["detection"]["confidence"] = args.conf

    if args.no_display:
        data["output"]["display"] = False

    return AppConfig.model_validate(data)


def handle_event(
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

def print_summary(result: PipelineResult) -> None:
    print()
    print("Processing completed")
    print(f"Frames: {result.processed_frames}")
    print(f"Source FPS: {result.source_fps:.2f}")
    print(f"Effective FPS: {result.effective_fps:.2f}")
    print(f"Core processing FPS: {result.core_processing_fps:.2f}")
    print(f"End-to-end FPS: {result.end_to_end_fps:.2f}")
    print(f"Elapsed time: {result.elapsed_seconds:.2f}s")
    print(f"IN: {result.in_count}")
    print(f"OUT: {result.out_count}")
    print(f"NET: {result.in_count - result.out_count}")
    print(f"Intrusions: {result.intrusion_count}")
    print(f"Loitering: {result.loitering_count}")
    print(f"Output: {result.output_path}")
    if result.stopped_early:
        print("Status: stopped early by user")
    else:
        print("Status: completed")

def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    logger = configure_logging(config.logging.level)
    source_path = Path(args.source)
    output_path = Path(args.output)
    config_data = config.model_dump(mode="json")
    run_id = str(uuid4())
    repository = SQLiteRepository(config.database.path)

    event_writer = None
    if config.logging.event_jsonl_path is not None:
        event_writer = EventJsonlWriter(
            event_path=config.logging.event_jsonl_path,
            metadata_path=config.logging.run_metadata_path,
            run_id=run_id,
        )

    recorder = None

    try:
        recorder = RunRecorder(
            repository=repository,
            run_id=run_id,
            source_path=source_path,
            output_path=output_path,
            config_data=config_data,
            event_writer=event_writer,
        )

        pipeline = VideoPipeline(
            config=config,
            event_handler=lambda event, frame_id: handle_event(
                event,
                frame_id,
                recorder,
                logger,
            ),
        )

        result = pipeline.run(
            source_path=source_path,
            output_path=output_path,
        )

        recorder.complete(
            result=result,
            config_data=config_data,
        )

        print_summary(result)
    except Exception as error:
        if recorder is not None:
            recorder.fail(error)

        logger.exception("Pipeline failed")
        raise
    finally:
        if recorder is not None:
            recorder.close()
        elif event_writer is not None:
            event_writer.close()

if __name__ == "__main__":
    main()
