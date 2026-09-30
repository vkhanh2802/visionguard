import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2

from src.config import AppConfig, load_config
from src.evaluation import (
    MotEvaluation,
    combine_evaluations,
    evaluate_mot_files,
    format_mot_prediction,
    load_sequence_info,
)
from src.tracking import TrackContinuityManager, YOLOByteTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VisionGuard tracking on MOT17 train sequences",
    )
    parser.add_argument("--dataset", default="MOT17", help="MOT17 dataset root")
    parser.add_argument(
        "--config",
        default="configs/week6.yaml",
        help="VisionGuard YAML config",
    )
    parser.add_argument(
        "--output-dir",
        default="data/mot17_benchmark",
        help="Prediction and report directory",
    )
    parser.add_argument(
        "--variant",
        choices=("DPM", "FRCNN", "SDP"),
        default="FRCNN",
        help="Use one detector variant because all variants share source images",
    )
    parser.add_argument(
        "--sequences",
        nargs="*",
        help="Optional base sequence numbers, for example 02 04 05",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for CLEAR and identity metrics",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override detection confidence from the YAML config",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override detection.model_path from the YAML config",
    )
    parser.add_argument(
        "--disable-continuity",
        action="store_true",
        help="Evaluate raw ByteTrack without canonical continuity",
    )
    parser.add_argument(
        "--reuse-results",
        action="store_true",
        help="Evaluate existing prediction files without running inference again",
    )
    return parser.parse_args()


def apply_benchmark_overrides(
    config: AppConfig,
    confidence: float | None,
    disable_continuity: bool,
    model_path: str | None = None,
) -> AppConfig:
    data = config.model_dump(mode="python")
    if model_path is not None:
        data["detection"]["model_path"] = model_path
    if confidence is not None:
        data["detection"]["confidence"] = confidence
    if disable_continuity:
        data["tracking"]["continuity_enabled"] = False
    return AppConfig.model_validate(data)


def create_continuity_manager(config: AppConfig) -> TrackContinuityManager:
    tracking = config.tracking
    return TrackContinuityManager(
        max_gap_frames=tracking.continuity_max_gap_frames,
        min_gap_frames=tracking.continuity_min_gap_frames,
        max_distance_px=tracking.continuity_max_distance_px,
        confirmation_frames=tracking.continuity_confirmation_frames,
        ambiguity_margin_px=tracking.continuity_ambiguity_margin_px,
        max_size_ratio=tracking.continuity_max_size_ratio,
    )


def discover_sequences(
    dataset_root: Path,
    variant: str,
    selected_sequences: list[str] | None,
) -> list[Path]:
    train_root = dataset_root / "train"
    sequence_paths = sorted(train_root.glob(f"MOT17-*-{variant}"))
    if selected_sequences:
        selected = {sequence.zfill(2) for sequence in selected_sequences}
        sequence_paths = [
            path
            for path in sequence_paths
            if path.name.split("-")[1] in selected
        ]
    if not sequence_paths:
        raise FileNotFoundError(
            f"No MOT17 {variant} train sequences found in {train_root}"
        )
    return sequence_paths


def run_sequence(
    sequence_path: Path,
    output_dir: Path,
    config: AppConfig,
) -> dict[str, object]:
    info = load_sequence_info(sequence_path / "seqinfo.ini")
    raw_path = output_dir / "raw" / f"{info.name}.txt"
    canonical_path = output_dir / "canonical" / f"{info.name}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = YOLOByteTracker(
        model_path=config.detection.model_path,
        confidence=config.detection.confidence,
        target_classes=config.detection.target_classes,
        tracker_config=config.tracking.tracker_config,
    )
    continuity = (
        create_continuity_manager(config)
        if config.tracking.continuity_enabled
        else None
    )
    image_dir = sequence_path / info.image_dir
    started_at = perf_counter()
    inference_seconds = 0.0

    with raw_path.open("w", encoding="utf-8", newline="") as raw_file, (
        canonical_path.open("w", encoding="utf-8", newline="")
    ) as canonical_file:
        for frame_id in range(1, info.length + 1):
            image_path = image_dir / f"{frame_id:06d}{info.image_extension}"
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read MOT17 frame: {image_path}")

            inference_started_at = perf_counter()
            raw_tracks = tracker.track(frame)
            inference_seconds += perf_counter() - inference_started_at
            canonical_tracks = (
                continuity.process(raw_tracks, frame_id - 1)
                if continuity is not None
                else raw_tracks
            )

            raw_file.writelines(
                format_mot_prediction(frame_id, track) for track in raw_tracks
            )
            canonical_file.writelines(
                format_mot_prediction(frame_id, track) for track in canonical_tracks
            )

    elapsed_seconds = perf_counter() - started_at
    return {
        "name": info.name,
        "frames": info.length,
        "frame_rate": info.frame_rate,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "end_to_end_fps": round(info.length / elapsed_seconds, 3),
        "inference_fps": round(info.length / inference_seconds, 3),
        "raw_path": str(raw_path),
        "canonical_path": str(canonical_path),
        "continuity": continuity.snapshot() if continuity is not None else None,
    }


def evaluate_sequence(
    sequence_path: Path,
    output_dir: Path,
    iou_threshold: float,
) -> tuple[MotEvaluation, MotEvaluation]:
    name = sequence_path.name
    ground_truth_path = sequence_path / "gt" / "gt.txt"
    raw_evaluation = evaluate_mot_files(
        ground_truth_path,
        output_dir / "raw" / f"{name}.txt",
        iou_threshold,
    )
    canonical_evaluation = evaluate_mot_files(
        ground_truth_path,
        output_dir / "canonical" / f"{name}.txt",
        iou_threshold,
    )
    return raw_evaluation, canonical_evaluation


def write_report(output_dir: Path, report: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    aggregate = report["aggregate"]
    raw = aggregate["raw"]
    canonical = aggregate["canonical"]
    markdown = [
        "# MOT17 Baseline\n",
        "| Output | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | FPS |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
        (
            f"| Raw ByteTrack | {raw['hota']:.3f} | {raw['deta']:.3f} | "
            f"{raw['assa']:.3f} | {raw['mota']:.3f} | {raw['idf1']:.3f} | "
            f"{raw['id_switches']} | {aggregate['end_to_end_fps']:.3f} |\n"
        ),
        (
            f"| Canonical continuity | {canonical['hota']:.3f} | "
            f"{canonical['deta']:.3f} | {canonical['assa']:.3f} | "
            f"{canonical['mota']:.3f} | {canonical['idf1']:.3f} | "
            f"{canonical['id_switches']} | {aggregate['end_to_end_fps']:.3f} |\n"
        ),
    ]
    (output_dir / "report.md").write_text("".join(markdown), encoding="utf-8")


def print_summary(report: dict[str, object]) -> None:
    aggregate = report["aggregate"]
    print()
    print("MOT17 baseline completed")
    print(
        "Output                 HOTA    DetA    AssA    MOTA    IDF1    IDSW"
    )
    for label, key in (
        ("Raw ByteTrack", "raw"),
        ("Canonical continuity", "canonical"),
    ):
        metrics = aggregate[key]
        print(
            f"{label:<22} "
            f"{metrics['hota']:>6.2f} "
            f"{metrics['deta']:>7.2f} "
            f"{metrics['assa']:>7.2f} "
            f"{metrics['mota']:>7.2f} "
            f"{metrics['idf1']:>7.2f} "
            f"{metrics['id_switches']:>7}"
        )
    print(f"End-to-end FPS: {aggregate['end_to_end_fps']:.2f}")
    print(f"Report: {report['report_path']}")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    output_dir = Path(args.output_dir)
    config = load_config(args.config)
    config = apply_benchmark_overrides(
        config,
        confidence=args.conf,
        disable_continuity=args.disable_continuity,
        model_path=args.model,
    )
    sequence_paths = discover_sequences(
        dataset_root,
        args.variant,
        args.sequences,
    )

    sequence_reports: list[dict[str, object]] = []
    raw_evaluations: list[MotEvaluation] = []
    canonical_evaluations: list[MotEvaluation] = []
    total_frames = 0
    total_elapsed_seconds = 0.0

    for sequence_path in sequence_paths:
        print(f"Processing {sequence_path.name}...")
        if args.reuse_results:
            info = load_sequence_info(sequence_path / "seqinfo.ini")
            sequence_report: dict[str, object] = {
                "name": info.name,
                "frames": info.length,
            }
        else:
            sequence_report = run_sequence(sequence_path, output_dir, config)

        raw_evaluation, canonical_evaluation = evaluate_sequence(
            sequence_path,
            output_dir,
            args.iou_threshold,
        )
        sequence_report["raw"] = raw_evaluation.metrics()
        sequence_report["canonical"] = canonical_evaluation.metrics()
        sequence_reports.append(sequence_report)
        raw_evaluations.append(raw_evaluation)
        canonical_evaluations.append(canonical_evaluation)
        total_frames += int(sequence_report["frames"])
        total_elapsed_seconds += float(sequence_report.get("elapsed_seconds", 0.0))

    aggregate_raw = combine_evaluations(raw_evaluations).metrics()
    aggregate_canonical = combine_evaluations(canonical_evaluations).metrics()
    aggregate_fps = (
        total_frames / total_elapsed_seconds if total_elapsed_seconds > 0 else 0.0
    )
    report: dict[str, object] = {
        "dataset": str(dataset_root),
        "config": str(args.config),
        "detector_model": config.detection.model_path,
        "variant": args.variant,
        "detector_confidence": config.detection.confidence,
        "continuity_enabled": config.tracking.continuity_enabled,
        "iou_threshold": args.iou_threshold,
        "sequences": sequence_reports,
        "aggregate": {
            "frames": total_frames,
            "elapsed_seconds": round(total_elapsed_seconds, 3),
            "end_to_end_fps": round(aggregate_fps, 3),
            "raw": aggregate_raw,
            "canonical": aggregate_canonical,
        },
        "report_path": str(output_dir / "report.json"),
    }
    write_report(output_dir, report)
    print_summary(report)


if __name__ == "__main__":
    main()
