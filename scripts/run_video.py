import argparse
from pathlib import Path

from src.database import SQLiteRepository
from src.logging_config import configure_logging

from src.analysis_service import create_analysis_job, execute_analysis_job
from src.config import AppConfig, load_config
from src.pipeline import PipelineResult


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
    repository = SQLiteRepository(config.database.path)
    job = create_analysis_job(
        config=config,
        source_path=source_path,
        output_path=output_path,
        repository=repository,
        logger=logger,
    )
    try:
        result = execute_analysis_job(job)
    except Exception:
        logger.exception("Pipeline failed")
        raise

    print_summary(result)

if __name__ == "__main__":
    main()
