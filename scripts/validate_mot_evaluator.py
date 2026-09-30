import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


METRIC_KEYS = (
    "hota",
    "deta",
    "assa",
    "loca",
    "mota",
    "motp",
    "idf1",
    "true_positives",
    "false_positives",
    "false_negatives",
    "id_switches",
    "fragmentations",
)
RESULT_PREFIX = "TRACKEVAL_RESULT="
TRACKEVAL_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare VisionGuard MOT metrics with official TrackEval metrics",
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--trackeval-root", required=True)
    parser.add_argument(
        "--trackeval-python",
        default=sys.executable,
        help="Python executable with NumPy and SciPy available",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", help="Optional JSON validation report path")
    parser.add_argument("--trackeval-worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def _mot_frame_count(*paths: Path) -> int:
    frame_count = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                columns = line.split(",", maxsplit=1)
                if len(columns) < 2:
                    raise ValueError(f"Invalid MOT row at {path}:{line_number}")
                frame_count = max(frame_count, int(float(columns[0])))
    return frame_count


def _describe_artifact(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _build_trackeval_data(trackeval, ground_truth_path: Path, prediction_path: Path):
    sequence_name = "VISIONGUARD"
    tracker_name = "visionguard"
    frame_count = _mot_frame_count(ground_truth_path, prediction_path)
    with TemporaryDirectory(prefix="visionguard-trackeval-") as temporary_directory:
        root = Path(temporary_directory)
        ground_truth_file = root / "gt" / sequence_name / "gt" / "gt.txt"
        prediction_file = (
            root / "trackers" / tracker_name / "data" / f"{sequence_name}.txt"
        )
        ground_truth_file.parent.mkdir(parents=True)
        prediction_file.parent.mkdir(parents=True)
        shutil.copyfile(ground_truth_path, ground_truth_file)
        shutil.copyfile(prediction_path, prediction_file)

        dataset = trackeval.datasets.MotChallenge2DBox(
            {
                "GT_FOLDER": str(root / "gt"),
                "TRACKERS_FOLDER": str(root / "trackers"),
                "OUTPUT_FOLDER": str(root / "results"),
                "TRACKERS_TO_EVAL": [tracker_name],
                "CLASSES_TO_EVAL": ["pedestrian"],
                "BENCHMARK": "MOT17",
                "SPLIT_TO_EVAL": "train",
                "SEQ_INFO": {sequence_name: frame_count},
                "SKIP_SPLIT_FOL": True,
                "PRINT_CONFIG": False,
            }
        )
        raw_data = dataset.get_raw_seq_data(tracker_name, sequence_name)
        return dataset.get_preprocessed_seq_data(raw_data, "pedestrian")


def _run_trackeval_worker(args: argparse.Namespace) -> None:
    import numpy as np

    # TrackEval currently uses aliases removed in NumPy 2.x.
    np.float = float
    np.int = int
    trackeval_root = Path(args.trackeval_root).resolve()
    sys.path.insert(0, str(trackeval_root))
    import trackeval

    imported_path = Path(trackeval.__file__).resolve()
    if not imported_path.is_relative_to(trackeval_root):
        raise RuntimeError(
            f"Imported TrackEval from {imported_path}, outside {trackeval_root}."
        )
    data = _build_trackeval_data(
        trackeval,
        Path(args.ground_truth),
        Path(args.predictions),
    )
    metric_config = {
        "THRESHOLD": args.iou_threshold,
        "PRINT_CONFIG": False,
    }
    hota = trackeval.metrics.HOTA().eval_sequence(data)
    clear = trackeval.metrics.CLEAR(metric_config).eval_sequence(data)
    identity = trackeval.metrics.Identity(metric_config).eval_sequence(data)
    result = {
        "hota": round(float(np.mean(hota["HOTA"])) * 100, 3),
        "deta": round(float(np.mean(hota["DetA"])) * 100, 3),
        "assa": round(float(np.mean(hota["AssA"])) * 100, 3),
        "loca": round(float(np.mean(hota["LocA"])) * 100, 3),
        "mota": round(float(clear["MOTA"]) * 100, 3),
        "motp": round(float(clear["MOTP"]) * 100, 3),
        "idf1": round(float(identity["IDF1"]) * 100, 3),
        "true_positives": int(clear["CLR_TP"]),
        "false_positives": int(clear["CLR_FP"]),
        "false_negatives": int(clear["CLR_FN"]),
        "id_switches": int(clear["IDSW"]),
        "fragmentations": int(clear["Frag"]),
    }
    print(f"{RESULT_PREFIX}{json.dumps(result, sort_keys=True)}")


def _run_official_trackeval(args: argparse.Namespace) -> dict[str, int | float]:
    command = [
        args.trackeval_python,
        str(Path(__file__).resolve()),
        "--trackeval-worker",
        "--ground-truth",
        str(Path(args.ground_truth).resolve()),
        "--predictions",
        str(Path(args.predictions).resolve()),
        "--trackeval-root",
        str(Path(args.trackeval_root).resolve()),
        "--iou-threshold",
        str(args.iou_threshold),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    result_line = next(
        (
            line
            for line in completed.stdout.splitlines()
            if line.startswith(RESULT_PREFIX)
        ),
        None,
    )
    if result_line is None:
        raise RuntimeError(
            "TrackEval worker did not return a result.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return json.loads(result_line.removeprefix(RESULT_PREFIX))


def _trackeval_checkout_metadata(root: Path) -> dict[str, object]:
    top_level = Path(
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    ).resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    package_path = root / "trackeval" / "__init__.py"
    if top_level != root or not package_path.is_file():
        raise RuntimeError(f"TrackEval root is not the repository root: {root}")
    if commit != TRACKEVAL_COMMIT or dirty:
        raise RuntimeError(
            "TrackEval checkout must be clean and pinned to "
            f"{TRACKEVAL_COMMIT}; found commit={commit}, dirty={dirty}."
        )
    return {"root": str(root), "commit": commit, "dirty": dirty}


def main() -> None:
    args = parse_args()
    if args.trackeval_worker:
        _run_trackeval_worker(args)
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.evaluation import evaluate_mot_files

    trackeval_metadata = _trackeval_checkout_metadata(
        Path(args.trackeval_root).resolve()
    )
    visionguard = evaluate_mot_files(
        Path(args.ground_truth),
        Path(args.predictions),
        iou_threshold=args.iou_threshold,
    ).metrics()
    official = _run_official_trackeval(args)
    differences = {
        key: round(float(visionguard[key]) - float(official[key]), 6)
        for key in METRIC_KEYS
        if visionguard[key] != official[key]
    }
    report = {
        "ground_truth": _describe_artifact(Path(args.ground_truth)),
        "predictions": _describe_artifact(Path(args.predictions)),
        "trackeval": trackeval_metadata,
        "iou_threshold": args.iou_threshold,
        "visionguard": {key: visionguard[key] for key in METRIC_KEYS},
        "trackeval_metrics": official,
        "differences": differences,
        "matches": not differences,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    if differences:
        raise SystemExit("VisionGuard metrics differ from TrackEval")


if __name__ == "__main__":
    main()
