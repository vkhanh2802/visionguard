import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2


CROWDDETECTION_COMMIT = "9786f58869a55af3e0b51fc78f8638a825dae4a2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Ultralytics predictions with the CrowdHuman protocol",
    )
    parser.add_argument("--ground-truth", required=True, help="CrowdHuman ODGT file")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Ultralytics COCO-format predictions.json",
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="CrowdHuman validation image directory",
    )
    parser.add_argument(
        "--evaluator-root",
        required=True,
        help="Pinned megvii-model/CrowdDetection checkout",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def describe_artifact(path: str | Path) -> dict[str, object]:
    artifact_path = Path(path)
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def load_odgt(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group_predictions(
    predictions: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for prediction in predictions:
        image_id = str(prediction["image_id"])
        box = [float(value) for value in prediction["bbox"]]
        score = float(prediction["score"])
        if len(box) != 4 or not all(math.isfinite(value) for value in [*box, score]):
            raise ValueError(f"Invalid prediction for image {image_id}")
        grouped.setdefault(image_id, []).append({"box": box, "score": score})
    return grouped


def build_detection_records(
    ground_truth_records: list[dict[str, object]],
    grouped_predictions: dict[str, list[dict[str, object]]],
    images_dir: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    ground_truth_ids = {str(record["ID"]) for record in ground_truth_records}
    unexpected_ids = sorted(set(grouped_predictions) - ground_truth_ids)
    if unexpected_ids:
        raise ValueError(
            f"Predictions contain {len(unexpected_ids)} unknown image IDs: "
            f"{unexpected_ids[:3]}"
        )

    records = []
    missing_images = []
    for ground_truth in ground_truth_records:
        image_id = str(ground_truth["ID"])
        image_path = images_dir / f"{image_id}.jpg"
        image = cv2.imread(str(image_path)) if image_path.is_file() else None
        if image is None:
            if grouped_predictions.get(image_id):
                raise FileNotFoundError(
                    f"Image is missing but has predictions: {image_path}"
                )
            missing_images.append(image_id)
            width = height = 1
        else:
            height, width = image.shape[:2]
        records.append(
            {
                "ID": image_id,
                "width": width,
                "height": height,
                "dtboxes": grouped_predictions.get(image_id, []),
            }
        )
    return records, missing_images


def write_odgt(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def evaluator_metadata(root: Path) -> dict[str, object]:
    resolved_root = root.resolve()
    package_path = resolved_root / "evaluate" / "APMRToolkits" / "__init__.py"
    if not package_path.is_file():
        raise RuntimeError(f"CrowdDetection evaluator not found: {resolved_root}")
    commit = subprocess.run(
        ["git", "-C", str(resolved_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "-C", str(resolved_root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    unexpected_changes = [
        line
        for line in status_lines
        if "__pycache__/" not in line and not line.endswith(".pyc")
    ]
    dirty = bool(unexpected_changes)
    if commit != CROWDDETECTION_COMMIT or dirty:
        raise RuntimeError(
            "CrowdDetection checkout must be clean and pinned to "
            f"{CROWDDETECTION_COMMIT}; found commit={commit}, dirty={dirty}."
        )
    return {"root": str(resolved_root), "commit": commit, "dirty": dirty}


def run_official_evaluation(
    evaluator_root: Path,
    ground_truth_path: Path,
    detections_path: Path,
) -> dict[str, float]:
    evaluate_path = evaluator_root.resolve() / "evaluate"
    sys.path.insert(0, str(evaluate_path))
    previous_bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        from APMRToolkits import Database
    finally:
        sys.dont_write_bytecode = previous_bytecode_setting
        sys.path.pop(0)

    database = Database(
        str(ground_truth_path),
        str(detections_path),
        "box",
        None,
        0,
    )
    database.compare()
    average_precision, _ = database.eval_AP()
    miss_rate, _ = database.eval_MR()
    return {
        "average_precision": round(float(average_precision), 6),
        "log_average_miss_rate": round(float(miss_rate), 6),
    }


def main() -> None:
    args = parse_args()
    ground_truth_path = Path(args.ground_truth)
    predictions_path = Path(args.predictions)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_records = load_odgt(ground_truth_path)
    with predictions_path.open(encoding="utf-8") as handle:
        predictions = json.load(handle)
    grouped_predictions = group_predictions(predictions)
    detection_records, missing_images = build_detection_records(
        ground_truth_records,
        grouped_predictions,
        images_dir,
    )
    detections_path = output_dir / "detections.odgt"
    write_odgt(detections_path, detection_records)

    evaluator = evaluator_metadata(Path(args.evaluator_root))
    metrics = run_official_evaluation(
        Path(args.evaluator_root),
        ground_truth_path,
        detections_path,
    )
    report = {
        "protocol": "CrowdHuman full-body Caltech matching",
        "metrics": metrics,
        "image_count": len(ground_truth_records),
        "images_with_predictions": len(grouped_predictions),
        "prediction_count": len(predictions),
        "missing_image_count": len(missing_images),
        "missing_images": missing_images,
        "evaluator": evaluator,
        "artifacts": {
            "ground_truth": describe_artifact(ground_truth_path),
            "predictions": describe_artifact(predictions_path),
            "detections": describe_artifact(detections_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"AP={metrics['average_precision']:.4f} "
        f"mMR={metrics['log_average_miss_rate']:.4f}"
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
