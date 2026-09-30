from collections import defaultdict
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

import lap
import numpy as np

from src.tracking import Track


BBox = tuple[float, float, float, float]
HOTA_THRESHOLDS = np.arange(0.05, 0.96, 0.05)
DISTRACTOR_CLASSES = {2, 7, 8, 12}


@dataclass(frozen=True)
class MotSequenceInfo:
    name: str
    image_dir: str
    frame_rate: float
    length: int
    width: int
    height: int
    image_extension: str


@dataclass(frozen=True)
class MotDetection:
    frame_id: int
    track_id: int
    bbox: BBox
    confidence: float = 1.0
    class_id: int = 1
    visibility: float = 1.0


@dataclass
class _FrameData:
    ground_truth: list[MotDetection]
    predictions: list[MotDetection]
    similarities: np.ndarray


@dataclass
class MotEvaluation:
    ground_truth_detections: int
    predicted_detections: int
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    fragmentations: int
    matching_iou_sum: float
    id_true_positives: int
    id_false_positives: int
    id_false_negatives: int
    hota_true_positives: np.ndarray
    hota_false_positives: np.ndarray
    hota_false_negatives: np.ndarray
    hota_association_numerator: np.ndarray
    hota_localization_sum: np.ndarray

    def metrics(self) -> dict[str, int | float]:
        mota = _safe_divide(
            self.ground_truth_detections
            - self.false_negatives
            - self.false_positives
            - self.id_switches,
            self.ground_truth_detections,
        )
        motp = _safe_divide(self.matching_iou_sum, self.true_positives)
        idf1 = _safe_divide(
            2 * self.id_true_positives,
            2 * self.id_true_positives
            + self.id_false_positives
            + self.id_false_negatives,
        )

        hota_denominator = (
            self.hota_true_positives
            + self.hota_false_positives
            + self.hota_false_negatives
        )
        detection_accuracy = np.divide(
            self.hota_true_positives,
            hota_denominator,
            out=np.zeros_like(self.hota_true_positives, dtype=float),
            where=hota_denominator > 0,
        )
        association_accuracy = np.divide(
            self.hota_association_numerator,
            self.hota_true_positives,
            out=np.zeros_like(self.hota_true_positives, dtype=float),
            where=self.hota_true_positives > 0,
        )
        localization_accuracy = np.divide(
            self.hota_localization_sum,
            self.hota_true_positives,
            out=np.zeros_like(self.hota_true_positives, dtype=float),
            where=self.hota_true_positives > 0,
        )
        hota = np.sqrt(detection_accuracy * association_accuracy)

        return {
            "ground_truth_detections": self.ground_truth_detections,
            "predicted_detections": self.predicted_detections,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "hota": round(float(np.mean(hota)) * 100, 3),
            "deta": round(float(np.mean(detection_accuracy)) * 100, 3),
            "assa": round(float(np.mean(association_accuracy)) * 100, 3),
            "loca": round(float(np.mean(localization_accuracy)) * 100, 3),
            "mota": round(mota * 100, 3),
            "motp": round(motp * 100, 3),
            "idf1": round(idf1 * 100, 3),
            "precision": round(
                _safe_divide(
                    self.true_positives,
                    self.true_positives + self.false_positives,
                )
                * 100,
                3,
            ),
            "recall": round(
                _safe_divide(
                    self.true_positives,
                    self.true_positives + self.false_negatives,
                )
                * 100,
                3,
            ),
        }


def load_sequence_info(path: Path) -> MotSequenceInfo:
    parser = ConfigParser()
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"MOT sequence metadata not found: {path}")
    if "Sequence" not in parser:
        raise ValueError(f"Missing [Sequence] section: {path}")

    sequence = parser["Sequence"]
    return MotSequenceInfo(
        name=sequence["name"],
        image_dir=sequence.get("imDir", "img1"),
        frame_rate=sequence.getfloat("frameRate"),
        length=sequence.getint("seqLength"),
        width=sequence.getint("imWidth"),
        height=sequence.getint("imHeight"),
        image_extension=sequence.get("imExt", ".jpg"),
    )


def format_mot_prediction(frame_id: int, track: Track) -> str:
    x1, y1, x2, y2 = track.bbox
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return (
        f"{frame_id},{track.track_id},{x1:.2f},{y1:.2f},"
        f"{width:.2f},{height:.2f},{track.confidence:.6f},-1,-1,-1\n"
    )


def evaluate_mot_files(
    ground_truth_path: Path,
    prediction_path: Path,
    iou_threshold: float = 0.5,
) -> MotEvaluation:
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1].")

    ground_truth, ignored = _load_ground_truth(ground_truth_path)
    predictions = _load_predictions(prediction_path)
    frames = _prepare_frames(
        ground_truth=ground_truth,
        ignored=ignored,
        predictions=predictions,
        iou_threshold=iou_threshold,
    )
    return _evaluate_frames(frames, iou_threshold)


def combine_evaluations(evaluations: list[MotEvaluation]) -> MotEvaluation:
    if not evaluations:
        raise ValueError("At least one evaluation is required.")

    return MotEvaluation(
        ground_truth_detections=sum(
            evaluation.ground_truth_detections for evaluation in evaluations
        ),
        predicted_detections=sum(
            evaluation.predicted_detections for evaluation in evaluations
        ),
        true_positives=sum(evaluation.true_positives for evaluation in evaluations),
        false_positives=sum(evaluation.false_positives for evaluation in evaluations),
        false_negatives=sum(evaluation.false_negatives for evaluation in evaluations),
        id_switches=sum(evaluation.id_switches for evaluation in evaluations),
        fragmentations=sum(evaluation.fragmentations for evaluation in evaluations),
        matching_iou_sum=sum(
            evaluation.matching_iou_sum for evaluation in evaluations
        ),
        id_true_positives=sum(
            evaluation.id_true_positives for evaluation in evaluations
        ),
        id_false_positives=sum(
            evaluation.id_false_positives for evaluation in evaluations
        ),
        id_false_negatives=sum(
            evaluation.id_false_negatives for evaluation in evaluations
        ),
        hota_true_positives=sum(
            (evaluation.hota_true_positives for evaluation in evaluations),
            start=np.zeros_like(HOTA_THRESHOLDS, dtype=int),
        ),
        hota_false_positives=sum(
            (evaluation.hota_false_positives for evaluation in evaluations),
            start=np.zeros_like(HOTA_THRESHOLDS, dtype=int),
        ),
        hota_false_negatives=sum(
            (evaluation.hota_false_negatives for evaluation in evaluations),
            start=np.zeros_like(HOTA_THRESHOLDS, dtype=int),
        ),
        hota_association_numerator=sum(
            (evaluation.hota_association_numerator for evaluation in evaluations),
            start=np.zeros_like(HOTA_THRESHOLDS, dtype=float),
        ),
        hota_localization_sum=sum(
            (evaluation.hota_localization_sum for evaluation in evaluations),
            start=np.zeros_like(HOTA_THRESHOLDS, dtype=float),
        ),
    )


def _load_ground_truth(
    path: Path,
) -> tuple[dict[int, list[MotDetection]], dict[int, list[MotDetection]]]:
    ground_truth: dict[int, list[MotDetection]] = defaultdict(list)
    ignored: dict[int, list[MotDetection]] = defaultdict(list)

    for detection in _read_mot_file(path):
        if detection.confidence > 0 and detection.class_id == 1:
            ground_truth[detection.frame_id].append(detection)
        elif detection.confidence <= 0 or detection.class_id in DISTRACTOR_CLASSES:
            ignored[detection.frame_id].append(detection)

    return dict(ground_truth), dict(ignored)


def _load_predictions(path: Path) -> dict[int, list[MotDetection]]:
    predictions: dict[int, list[MotDetection]] = defaultdict(list)
    for detection in _read_mot_file(path, prediction=True):
        predictions[detection.frame_id].append(detection)
    return dict(predictions)


def _read_mot_file(path: Path, prediction: bool = False) -> list[MotDetection]:
    detections: list[MotDetection] = []
    if not path.exists():
        raise FileNotFoundError(f"MOT file not found: {path}")

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            columns = stripped.split(",")
            if len(columns) < 7:
                raise ValueError(f"Invalid MOT row at {path}:{line_number}")

            x, y, width, height = (float(value) for value in columns[2:6])
            class_id = 1 if prediction or len(columns) < 8 else int(float(columns[7]))
            visibility = 1.0 if prediction or len(columns) < 9 else float(columns[8])
            detections.append(
                MotDetection(
                    frame_id=int(float(columns[0])),
                    track_id=int(float(columns[1])),
                    bbox=(x, y, width, height),
                    confidence=float(columns[6]),
                    class_id=class_id,
                    visibility=visibility,
                )
            )

    return detections


def _prepare_frames(
    ground_truth: dict[int, list[MotDetection]],
    ignored: dict[int, list[MotDetection]],
    predictions: dict[int, list[MotDetection]],
    iou_threshold: float,
) -> list[_FrameData]:
    frames: list[_FrameData] = []
    frame_ids = sorted(set(ground_truth) | set(ignored) | set(predictions))

    for frame_id in frame_ids:
        frame_ground_truth = ground_truth.get(frame_id, [])
        frame_predictions = predictions.get(frame_id, [])
        similarities = _iou_matrix(frame_ground_truth, frame_predictions)
        _, matched_prediction_indices = _match_similarities(
            similarities,
            iou_threshold,
        )
        matched_predictions = set(matched_prediction_indices.tolist())
        ignored_boxes = ignored.get(frame_id, [])
        ignored_overlaps = _ioa_matrix(ignored_boxes, frame_predictions)

        keep_indices = []
        for prediction_index in range(len(frame_predictions)):
            if prediction_index in matched_predictions:
                keep_indices.append(prediction_index)
                continue
            if (
                ignored_overlaps.size > 0
                and np.max(ignored_overlaps[:, prediction_index]) >= 0.5
            ):
                continue
            keep_indices.append(prediction_index)

        filtered_predictions = [frame_predictions[index] for index in keep_indices]
        frames.append(
            _FrameData(
                ground_truth=frame_ground_truth,
                predictions=filtered_predictions,
                similarities=_iou_matrix(frame_ground_truth, filtered_predictions),
            )
        )

    return frames


def _evaluate_frames(frames: list[_FrameData], iou_threshold: float) -> MotEvaluation:
    ground_truth_count = sum(len(frame.ground_truth) for frame in frames)
    prediction_count = sum(len(frame.predictions) for frame in frames)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    id_switches = 0
    fragmentations = 0
    matching_iou_sum = 0.0
    last_match: dict[int, int] = {}
    previous_matched: dict[int, bool] = {}
    ever_matched: set[int] = set()
    identity_matches: dict[tuple[int, int], int] = defaultdict(int)

    for frame in frames:
        continuity_scores = frame.similarities.copy()
        for ground_truth_index, detection in enumerate(frame.ground_truth):
            previous_id = last_match.get(detection.track_id)
            if previous_id is None:
                continue
            for prediction_index, prediction in enumerate(frame.predictions):
                if (
                    prediction.track_id == previous_id
                    and continuity_scores[ground_truth_index, prediction_index]
                    >= iou_threshold
                ):
                    continuity_scores[ground_truth_index, prediction_index] += 1000.0

        matched_ground_truth, matched_predictions = _match_similarities(
            continuity_scores,
            iou_threshold,
            threshold_values=frame.similarities,
        )
        true_positives += len(matched_ground_truth)
        false_negatives += len(frame.ground_truth) - len(matched_ground_truth)
        false_positives += len(frame.predictions) - len(matched_predictions)
        if len(matched_ground_truth) > 0:
            matching_iou_sum += float(
                frame.similarities[matched_ground_truth, matched_predictions].sum()
            )

        matched_ids: set[int] = set()
        for ground_truth_index, prediction_index in zip(
            matched_ground_truth,
            matched_predictions,
        ):
            ground_truth_id = frame.ground_truth[ground_truth_index].track_id
            prediction_id = frame.predictions[prediction_index].track_id
            previous_id = last_match.get(ground_truth_id)
            if previous_id is not None and previous_id != prediction_id:
                id_switches += 1
            if ground_truth_id in ever_matched and not previous_matched.get(
                ground_truth_id,
                False,
            ):
                fragmentations += 1
            last_match[ground_truth_id] = prediction_id
            previous_matched[ground_truth_id] = True
            ever_matched.add(ground_truth_id)
            matched_ids.add(ground_truth_id)

        for detection in frame.ground_truth:
            if detection.track_id not in matched_ids:
                previous_matched[detection.track_id] = False

        identity_ground_truth, identity_predictions = _match_similarities(
            frame.similarities,
            iou_threshold,
        )
        for ground_truth_index, prediction_index in zip(
            identity_ground_truth,
            identity_predictions,
        ):
            identity_matches[
                (
                    frame.ground_truth[ground_truth_index].track_id,
                    frame.predictions[prediction_index].track_id,
                )
            ] += 1

    id_true_positives = _maximum_identity_matches(identity_matches)
    hota = _calculate_hota(frames)
    return MotEvaluation(
        ground_truth_detections=ground_truth_count,
        predicted_detections=prediction_count,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        id_switches=id_switches,
        fragmentations=fragmentations,
        matching_iou_sum=matching_iou_sum,
        id_true_positives=id_true_positives,
        id_false_positives=prediction_count - id_true_positives,
        id_false_negatives=ground_truth_count - id_true_positives,
        hota_true_positives=hota[0],
        hota_false_positives=hota[1],
        hota_false_negatives=hota[2],
        hota_association_numerator=hota[3],
        hota_localization_sum=hota[4],
    )


def _calculate_hota(
    frames: list[_FrameData],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ground_truth_ids = sorted(
        {detection.track_id for frame in frames for detection in frame.ground_truth}
    )
    prediction_ids = sorted(
        {detection.track_id for frame in frames for detection in frame.predictions}
    )
    ground_truth_lookup = {
        track_id: index for index, track_id in enumerate(ground_truth_ids)
    }
    prediction_lookup = {
        track_id: index for index, track_id in enumerate(prediction_ids)
    }
    ground_truth_id_count = np.zeros(len(ground_truth_ids), dtype=float)
    prediction_id_count = np.zeros(len(prediction_ids), dtype=float)
    potential_matches = np.zeros(
        (len(ground_truth_ids), len(prediction_ids)),
        dtype=float,
    )

    indexed_frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for frame in frames:
        frame_ground_truth_ids = np.array(
            [ground_truth_lookup[item.track_id] for item in frame.ground_truth],
            dtype=int,
        )
        frame_prediction_ids = np.array(
            [prediction_lookup[item.track_id] for item in frame.predictions],
            dtype=int,
        )
        indexed_frames.append(
            (frame_ground_truth_ids, frame_prediction_ids, frame.similarities)
        )
        if len(frame_ground_truth_ids) > 0:
            ground_truth_id_count[frame_ground_truth_ids] += 1
        if len(frame_prediction_ids) > 0:
            prediction_id_count[frame_prediction_ids] += 1
        if frame.similarities.size == 0:
            continue

        denominator = (
            frame.similarities.sum(axis=0)[None, :]
            + frame.similarities.sum(axis=1)[:, None]
            - frame.similarities
        )
        normalized_similarity = np.divide(
            frame.similarities,
            denominator,
            out=np.zeros_like(frame.similarities),
            where=frame.similarities > 0,
        )
        potential_matches[
            np.ix_(frame_ground_truth_ids, frame_prediction_ids)
        ] += normalized_similarity

    association_denominator = (
        ground_truth_id_count[:, None]
        + prediction_id_count[None, :]
        - potential_matches
    )
    global_alignment = np.divide(
        potential_matches,
        association_denominator,
        out=np.zeros_like(potential_matches),
        where=association_denominator > 0,
    )

    true_positives = np.zeros_like(HOTA_THRESHOLDS, dtype=int)
    false_positives = np.zeros_like(HOTA_THRESHOLDS, dtype=int)
    false_negatives = np.zeros_like(HOTA_THRESHOLDS, dtype=int)
    association_numerator = np.zeros_like(HOTA_THRESHOLDS, dtype=float)
    localization_sum = np.zeros_like(HOTA_THRESHOLDS, dtype=float)

    for threshold_index, threshold in enumerate(HOTA_THRESHOLDS):
        match_counts = np.zeros_like(potential_matches, dtype=float)
        for frame_ground_truth_ids, frame_prediction_ids, similarities in indexed_frames:
            if similarities.size == 0:
                false_negatives[threshold_index] += len(frame_ground_truth_ids)
                false_positives[threshold_index] += len(frame_prediction_ids)
                continue

            scores = (
                global_alignment[
                    np.ix_(frame_ground_truth_ids, frame_prediction_ids)
                ]
                * similarities
            )
            matched_ground_truth, matched_predictions = _match_similarities(
                scores,
                threshold,
                threshold_values=similarities,
            )
            match_count = len(matched_ground_truth)
            true_positives[threshold_index] += match_count
            false_negatives[threshold_index] += (
                len(frame_ground_truth_ids) - match_count
            )
            false_positives[threshold_index] += len(frame_prediction_ids) - match_count
            if match_count == 0:
                continue

            localization_sum[threshold_index] += float(
                similarities[matched_ground_truth, matched_predictions].sum()
            )
            np.add.at(
                match_counts,
                (
                    frame_ground_truth_ids[matched_ground_truth],
                    frame_prediction_ids[matched_predictions],
                ),
                1,
            )

        association_union = (
            ground_truth_id_count[:, None]
            + prediction_id_count[None, :]
            - match_counts
        )
        association_scores = np.divide(
            match_counts,
            association_union,
            out=np.zeros_like(match_counts),
            where=association_union > 0,
        )
        association_numerator[threshold_index] = float(
            np.sum(match_counts * association_scores)
        )

    return (
        true_positives,
        false_positives,
        false_negatives,
        association_numerator,
        localization_sum,
    )


def _maximum_identity_matches(matches: dict[tuple[int, int], int]) -> int:
    if not matches:
        return 0
    ground_truth_ids = sorted({key[0] for key in matches})
    prediction_ids = sorted({key[1] for key in matches})
    ground_truth_lookup = {
        track_id: index for index, track_id in enumerate(ground_truth_ids)
    }
    prediction_lookup = {
        track_id: index for index, track_id in enumerate(prediction_ids)
    }
    scores = np.zeros((len(ground_truth_ids), len(prediction_ids)), dtype=float)
    for (ground_truth_id, prediction_id), count in matches.items():
        scores[
            ground_truth_lookup[ground_truth_id],
            prediction_lookup[prediction_id],
        ] = count
    rows, columns = _maximize_assignment(scores)
    return int(scores[rows, columns].sum()) if len(rows) > 0 else 0


def _match_similarities(
    scores: np.ndarray,
    threshold: float,
    threshold_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if scores.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    threshold_values = scores if threshold_values is None else threshold_values
    eligible_scores = np.where(threshold_values >= threshold, scores, 0.0)
    rows, columns = _maximize_assignment(eligible_scores)
    valid = threshold_values[rows, columns] >= threshold
    return rows[valid], columns[valid]


def _maximize_assignment(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if scores.size == 0 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    _, assignment, _ = lap.lapjv(-scores, extend_cost=True)
    rows = np.flatnonzero(assignment >= 0)
    return rows, assignment[rows]


def _iou_matrix(
    ground_truth: list[MotDetection],
    predictions: list[MotDetection],
) -> np.ndarray:
    return _overlap_matrix(ground_truth, predictions, denominator="union")


def _ioa_matrix(
    ignored: list[MotDetection],
    predictions: list[MotDetection],
) -> np.ndarray:
    return _overlap_matrix(ignored, predictions, denominator="prediction")


def _overlap_matrix(
    left: list[MotDetection],
    right: list[MotDetection],
    denominator: str,
) -> np.ndarray:
    if not left or not right:
        return np.zeros((len(left), len(right)), dtype=float)

    left_boxes = np.array([detection.bbox for detection in left], dtype=float)
    right_boxes = np.array([detection.bbox for detection in right], dtype=float)
    left_x2 = left_boxes[:, 0] + left_boxes[:, 2]
    left_y2 = left_boxes[:, 1] + left_boxes[:, 3]
    right_x2 = right_boxes[:, 0] + right_boxes[:, 2]
    right_y2 = right_boxes[:, 1] + right_boxes[:, 3]
    intersection_width = np.maximum(
        0.0,
        np.minimum(left_x2[:, None], right_x2[None, :])
        - np.maximum(left_boxes[:, 0, None], right_boxes[None, :, 0]),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(left_y2[:, None], right_y2[None, :])
        - np.maximum(left_boxes[:, 1, None], right_boxes[None, :, 1]),
    )
    intersection = intersection_width * intersection_height
    left_area = left_boxes[:, 2] * left_boxes[:, 3]
    right_area = right_boxes[:, 2] * right_boxes[:, 3]
    if denominator == "prediction":
        area = right_area[None, :]
    else:
        area = left_area[:, None] + right_area[None, :] - intersection
    return np.divide(
        intersection,
        area,
        out=np.zeros_like(intersection),
        where=area > 0,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
