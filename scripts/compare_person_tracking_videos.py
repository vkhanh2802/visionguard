import argparse
import json
import re
from pathlib import Path

from src.config import AppConfig, load_config
from src.pipeline import PipelineResult, VideoPipeline


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two person detectors on unannotated camera videos",
    )
    parser.add_argument(
        "--video-dir",
        default="datasets/person_tracking/camera",
        help="Directory containing camera videos",
    )
    parser.add_argument(
        "--config",
        default="C:/VisionGuard/configs/week6.yaml",
        help="Base config used for detector and ByteTrack settings",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/person_tracking_ab",
        help="Annotated videos and report directory",
    )
    parser.add_argument(
        "--baseline-model",
        default="yolo26n.pt",
        help="Baseline detector checkpoint",
    )
    parser.add_argument(
        "--candidate-model",
        default="runs/person_detection/mot17_a/weights/best.pt",
        help="Candidate detector checkpoint",
    )
    parser.add_argument(
        "--candidate-label",
        default="mot17_a",
        help="Label used for the candidate in reports and output file names",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline model for final regression",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Fallback detection confidence used by both models",
    )
    parser.add_argument(
        "--baseline-confidence",
        type=float,
        help="Optional baseline confidence override",
    )
    parser.add_argument(
        "--candidate-confidence",
        type=float,
        help="Optional candidate confidence override",
    )
    parser.add_argument(
        "--videos",
        nargs="*",
        help="Optional video file names to run, for example 1.mp4 2.mp4",
    )
    return parser.parse_args()


def discover_videos(video_dir: Path, selected: list[str] | None) -> list[Path]:
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    selected_names = set(selected or [])
    videos = sorted(
        path
        for path in video_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
        and (not selected_names or path.name in selected_names)
    )
    if not videos:
        raise FileNotFoundError(f"No selected camera videos found in {video_dir}")
    return videos


def select_models(
    baseline_model: str,
    candidate_model: str,
    baseline_only: bool,
    candidate_label: str = "mot17_a",
) -> tuple[tuple[str, str], ...]:
    if (
        candidate_label == "baseline"
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", candidate_label) is None
    ):
        raise ValueError(
            "--candidate-label must be a file-safe label other than 'baseline'"
        )
    models = [("baseline", baseline_model)]
    if not baseline_only:
        models.append((candidate_label, candidate_model))
    return tuple(models)


def resolve_confidences(
    shared_confidence: float,
    baseline_confidence: float | None,
    candidate_confidence: float | None,
) -> tuple[float, float]:
    baseline = baseline_confidence if baseline_confidence is not None else shared_confidence
    candidate = (
        candidate_confidence if candidate_confidence is not None else shared_confidence
    )
    if any(not 0 < confidence < 1 for confidence in (baseline, candidate)):
        raise ValueError("Every confidence must be in (0, 1)")
    return baseline, candidate


def build_tracking_only_config(
    base_config: AppConfig,
    model_path: str,
    confidence: float,
) -> AppConfig:
    data = base_config.model_dump(mode="python")
    data["detection"]["model_path"] = model_path
    data["detection"]["confidence"] = confidence
    data["tracking"]["continuity_enabled"] = False
    data["tracking"]["duplicate_suppression_enabled"] = False
    data["events"]["line_crossing"]["enabled"] = False
    data["events"]["intrusion"]["enabled"] = False
    data["events"]["loitering"]["enabled"] = False
    data["events"]["zones"] = {}
    data["output"]["display"] = False
    data["logging"]["event_jsonl_path"] = None
    data["logging"]["run_metadata_path"] = None
    return AppConfig.model_validate(data)


def summarize_result(
    result: PipelineResult,
    video_name: str,
    model_label: str,
    model_path: str,
    confidence: float,
) -> dict[str, object]:
    diagnostics = result.tracking_diagnostics
    timing = diagnostics.get("timing", {})
    track_lifetimes = diagnostics.get("track_lifetimes", [])
    observed_track_frames = sum(
        int(track["observed_frames"]) for track in track_lifetimes
    )
    total_tracks = int(diagnostics.get("total_track_count", 0))
    tracks_with_gaps = int(diagnostics.get("tracks_with_gaps", 0))

    return {
        "video": video_name,
        "model": model_label,
        "model_path": model_path,
        "confidence": confidence,
        "output_path": str(result.output_path),
        "frames": result.processed_frames,
        "source_fps": round(result.source_fps, 3),
        "end_to_end_fps": round(result.end_to_end_fps, 3),
        "core_processing_fps": round(result.core_processing_fps, 3),
        "tracking_fps": timing.get("tracking_fps"),
        "timing": timing,
        "total_tracks": total_tracks,
        "observed_track_frames": observed_track_frames,
        "average_active_tracks": round(
            observed_track_frames / result.processed_frames,
            3,
        ),
        "track_churn_per_1000_observations": round(
            total_tracks * 1000 / observed_track_frames,
            3,
        )
        if observed_track_frames
        else None,
        "tracks_with_gaps": tracks_with_gaps,
        "tracks_with_gaps_percent": round(
            tracks_with_gaps * 100 / total_tracks,
            3,
        )
        if total_tracks
        else None,
        "total_missing_frames": int(diagnostics.get("total_missing_frames", 0)),
        "max_gap_frames": int(diagnostics.get("max_gap_frames", 0)),
        "median_observed_frames": diagnostics.get("median_observed_frames"),
    }


def write_report(output_dir: Path, report: dict[str, object]) -> None:
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = [
        (
            "# Person Tracking Final Regression\n"
            if report["mode"] == "baseline_only"
            else "# Person Tracking Camera A/B\n"
        ),
        "The values below are runtime diagnostics and do not use ground truth. "
        "See the camera benchmark reports for accuracy metrics.\n\n",
        "| Video | Model | Conf | Frames | Avg active | Tracks | Churn/1k obs | "
        "Tracks with gaps | Max gap | Tracking FPS | Core FPS | E2E FPS |\n",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: |\n",
    ]
    for run in report["runs"]:
        rows.append(
            f"| {run['video']} | {run['model']} | {run['confidence']:.2f} | "
            f"{run['frames']} | "
            f"{run['average_active_tracks']:.3f} | {run['total_tracks']} | "
            f"{run['track_churn_per_1000_observations']:.3f} | "
            f"{run['tracks_with_gaps']} | {run['max_gap_frames']} | "
            f"{run['tracking_fps']:.3f} | {run['core_processing_fps']:.3f} | "
            f"{run['end_to_end_fps']:.3f} |\n"
        )
    (output_dir / "report.md").write_text("".join(rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = discover_videos(video_dir, args.videos)
    base_config = load_config(args.config)
    baseline_confidence, candidate_confidence = resolve_confidences(
        shared_confidence=args.conf,
        baseline_confidence=args.baseline_confidence,
        candidate_confidence=args.candidate_confidence,
    )
    models = select_models(
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        baseline_only=args.baseline_only,
        candidate_label=args.candidate_label,
    )
    confidence_by_model = {
        "baseline": baseline_confidence,
        args.candidate_label: candidate_confidence,
    }

    pipelines: dict[str, VideoPipeline] = {}
    runs: list[dict[str, object]] = []
    for video_path in videos:
        for model_label, model_path in models:
            confidence = confidence_by_model[model_label]
            output_path = output_dir / f"{video_path.stem}_{model_label}.mp4"
            config = build_tracking_only_config(
                base_config,
                model_path=model_path,
                confidence=confidence,
            )
            print(f"Processing {video_path.name} with {model_label}...")
            pipeline = pipelines.setdefault(model_label, VideoPipeline(config))
            result = pipeline.run(video_path, output_path)
            summary = summarize_result(
                result,
                video_name=video_path.name,
                model_label=model_label,
                model_path=model_path,
                confidence=confidence,
            )
            runs.append(summary)
            print(
                f"  tracks={summary['total_tracks']} "
                f"avg_active={summary['average_active_tracks']} "
                f"fps={summary['end_to_end_fps']}"
            )

    report: dict[str, object] = {
        "video_dir": str(video_dir),
        "base_config": str(args.config),
        "confidence": (
            baseline_confidence
            if baseline_confidence == candidate_confidence
            else None
        ),
        "baseline_confidence": baseline_confidence,
        "candidate_confidence": candidate_confidence,
        "continuity_enabled": False,
        "duplicate_suppression_enabled": False,
        "events_enabled": False,
        "ground_truth_evaluated": False,
        "mode": "baseline_only" if args.baseline_only else "comparison",
        "runs": runs,
    }
    write_report(output_dir, report)
    print(f"Report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
