import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2

from src.config import load_config
from src.evaluation import evaluate_mot_files, format_mot_prediction
from src.tracking import YOLOByteTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate person tracking models on an annotated camera video",
    )
    parser.add_argument("--video", required=True, help="Annotated source video")
    parser.add_argument("--ground-truth", required=True, help="MOT gt.txt path")
    parser.add_argument(
        "--config",
        default="C:/VisionGuard/configs/week6.yaml",
        help="Base VisionGuard config for target classes and ByteTrack settings",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/camera_tracking_benchmark",
        help="Prediction and report directory",
    )
    parser.add_argument("--baseline-model", default="yolo26n.pt")
    parser.add_argument(
        "--candidate-model",
        default="runs/person_detection/mot17_a/weights/best.pt",
    )
    parser.add_argument(
        "--confidences",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5],
    )
    parser.add_argument(
        "--baseline-confidence",
        type=float,
        help="Locked baseline confidence; requires --candidate-confidence",
    )
    parser.add_argument(
        "--candidate-confidence",
        type=float,
        help="Locked candidate confidence; requires --baseline-confidence",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def run_tracking(
    video_path: Path,
    prediction_path: Path,
    model_path: str,
    confidence: float,
    target_classes: list[str],
    tracker_config: Path,
) -> dict[str, int | float]:
    tracker = YOLOByteTracker(
        model_path=model_path,
        confidence=confidence,
        target_classes=target_classes,
        tracker_config=tracker_config,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    frame_id = 0
    prediction_count = 0
    started_at = perf_counter()
    inference_seconds = 0.0
    with prediction_path.open("w", encoding="utf-8", newline="") as output:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            inference_started_at = perf_counter()
            tracks = tracker.track(frame)
            inference_seconds += perf_counter() - inference_started_at
            prediction_count += len(tracks)
            output.writelines(format_mot_prediction(frame_id, track) for track in tracks)
    capture.release()
    elapsed_seconds = perf_counter() - started_at
    return {
        "frames": frame_id,
        "predictions": prediction_count,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "end_to_end_fps": round(frame_id / elapsed_seconds, 3),
        "inference_fps": round(frame_id / inference_seconds, 3),
    }


def select_best_runs(runs: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best = {}
    for run in runs:
        model = str(run["model"])
        current = best.get(model)
        score = (run["hota"], run["idf1"], run["mota"])
        if current is None or score > (
            current["hota"],
            current["idf1"],
            current["mota"],
        ):
            best[model] = run
    return best


def build_run_specs(
    baseline_model: str,
    candidate_model: str,
    confidences: list[float],
    baseline_confidence: float | None,
    candidate_confidence: float | None,
) -> tuple[str, list[tuple[str, str, float]]]:
    if (baseline_confidence is None) != (candidate_confidence is None):
        raise ValueError(
            "--baseline-confidence and --candidate-confidence must be used together"
        )
    if baseline_confidence is not None and candidate_confidence is not None:
        return (
            "locked_holdout",
            [
                ("baseline", baseline_model, baseline_confidence),
                ("mot17_a", candidate_model, candidate_confidence),
            ],
        )
    return (
        "calibration_sweep",
        [
            (model_label, model_path, confidence)
            for model_label, model_path in (
                ("baseline", baseline_model),
                ("mot17_a", candidate_model),
            )
            for confidence in confidences
        ],
    )


def write_report(output_dir: Path, report: dict[str, object]) -> None:
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    rows = [
        "# Camera Tracking Ground-Truth Benchmark\n",
        "| Model | Confidence | HOTA | DetA | AssA | MOTA | IDF1 | "
        "Precision | Recall | IDSW | FPS |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: |\n",
    ]
    for run in report["runs"]:
        rows.append(
            f"| {run['model']} | {run['confidence']:.2f} | {run['hota']:.3f} | "
            f"{run['deta']:.3f} | {run['assa']:.3f} | {run['mota']:.3f} | "
            f"{run['idf1']:.3f} | {run['precision']:.3f} | "
            f"{run['recall']:.3f} | {run['id_switches']} | "
            f"{run['end_to_end_fps']:.3f} |\n"
        )
    (output_dir / "report.md").write_text("".join(rows), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("--iou-threshold must be in (0, 1]")
    selected_confidences = [
        *args.confidences,
        *(
            [args.baseline_confidence]
            if args.baseline_confidence is not None
            else []
        ),
        *(
            [args.candidate_confidence]
            if args.candidate_confidence is not None
            else []
        ),
    ]
    if any(not 0 < confidence < 1 for confidence in selected_confidences):
        raise ValueError("Every confidence must be in (0, 1)")

    video_path = Path(args.video)
    ground_truth_path = Path(args.ground_truth)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    evaluation_mode, run_specs = build_run_specs(
        baseline_model=args.baseline_model,
        candidate_model=args.candidate_model,
        confidences=args.confidences,
        baseline_confidence=args.baseline_confidence,
        candidate_confidence=args.candidate_confidence,
    )

    runs: list[dict[str, object]] = []
    for model_label, model_path, confidence in run_specs:
        run_name = f"{model_label}_conf_{confidence:.2f}".replace(".", "p")
        prediction_path = output_dir / "predictions" / f"{run_name}.txt"
        print(f"Processing {model_label} at confidence {confidence:.2f}...")
        runtime = run_tracking(
            video_path=video_path,
            prediction_path=prediction_path,
            model_path=model_path,
            confidence=confidence,
            target_classes=config.detection.target_classes,
            tracker_config=config.tracking.tracker_config,
        )
        metrics = evaluate_mot_files(
            ground_truth_path,
            prediction_path,
            iou_threshold=args.iou_threshold,
        ).metrics()
        run: dict[str, object] = {
            "model": model_label,
            "model_path": model_path,
            "confidence": confidence,
            "prediction_path": str(prediction_path),
            **runtime,
            **metrics,
        }
        runs.append(run)
        print(
            f"  HOTA={run['hota']:.3f} IDF1={run['idf1']:.3f} "
            f"MOTA={run['mota']:.3f}"
        )

    report: dict[str, object] = {
        "video": str(video_path),
        "ground_truth": str(ground_truth_path),
        "config": str(args.config),
        "tracker_config": str(config.tracking.tracker_config),
        "iou_threshold": args.iou_threshold,
        "evaluation_mode": evaluation_mode,
        "ground_truth_available": True,
        "runs": runs,
        "best_by_model": select_best_runs(runs),
    }
    write_report(output_dir, report)
    print(f"Report: {output_dir / 'report.json'}")


if __name__ == "__main__":
    main()
