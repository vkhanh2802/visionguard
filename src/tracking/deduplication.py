from dataclasses import dataclass
from math import dist

from src.tracking.types import Track
from src.utils.geometry import calculate_iou


@dataclass
class _ActiveState:
    track: Track
    previous_track: Track | None
    first_seen_frame: int
    last_seen_frame: int
    previous_seen_frame: int | None
    observed_frames: int


@dataclass
class _PendingDuplicate:
    primary_id: int
    duplicate_id: int
    first_frame: int
    last_frame: int
    consecutive_frames: int
    containment_ratio: float
    iou: float
    bottom_distance_px: float


@dataclass
class _DuplicateAlias:
    primary_id: int
    last_confirmed_frame: int


class ActiveTrackDeduplicator:
    """Quarantine a second overlapping box during conservative confirmation.

    A candidate is kept out of event input while its consecutive observations are
    confirmed. Both boxes remain available to visualization and raw diagnostics so a
    suppression decision is auditable.
    """

    def __init__(
        self,
        confirmation_frames: int = 3,
        min_containment_ratio: float = 0.85,
        max_bottom_distance_px: float = 35.0,
        max_area_ratio: float = 3.0,
        max_motion_difference_px: float = 12.0,
        state_retention_frames: int = 5,
        max_handoff_gap_frames: int = 5,
    ):
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1.")
        if not 0.0 < min_containment_ratio <= 1.0:
            raise ValueError("min_containment_ratio must be in (0, 1].")
        if max_bottom_distance_px <= 0:
            raise ValueError("max_bottom_distance_px must be positive.")
        if max_area_ratio < 1.0:
            raise ValueError("max_area_ratio must be at least 1.")
        if max_motion_difference_px < 0:
            raise ValueError("max_motion_difference_px must be non-negative.")
        if state_retention_frames < 0:
            raise ValueError("state_retention_frames must be non-negative.")
        if max_handoff_gap_frames < 0:
            raise ValueError("max_handoff_gap_frames must be non-negative.")

        self.confirmation_frames = confirmation_frames
        self.min_containment_ratio = min_containment_ratio
        self.max_bottom_distance_px = max_bottom_distance_px
        self.max_area_ratio = max_area_ratio
        self.max_motion_difference_px = max_motion_difference_px
        self.state_retention_frames = state_retention_frames
        self.max_handoff_gap_frames = max_handoff_gap_frames

        self._states: dict[int, _ActiveState] = {}
        self._pending: dict[tuple[int, int], _PendingDuplicate] = {}
        self._aliases: dict[int, _DuplicateAlias] = {}
        self._suppressed_observation_count = 0
        self._pending_duplicate_observation_count = 0
        self._ambiguous_pair_count = 0
        self._handoff_count = 0
        self._confirmed_pairs: list[dict[str, int | float]] = []

    def process(self, tracks: list[Track], frame_id: int) -> list[Track]:
        self._cleanup(frame_id)
        current_ids = {track.track_id for track in tracks}
        for track in tracks:
            self._observe(track, frame_id)

        candidates_by_duplicate: dict[int, list[_PendingDuplicate]] = {}
        for index, left in enumerate(tracks):
            for right in tracks[index + 1 :]:
                candidate = self._candidate(left, right, frame_id)
                if candidate is not None:
                    candidates_by_duplicate.setdefault(
                        candidate.duplicate_id,
                        [],
                    ).append(candidate)

        current_pending: dict[tuple[int, int], _PendingDuplicate] = {}
        suppress_ids: set[int] = set()

        for duplicate_id, candidates in candidates_by_duplicate.items():
            if len(candidates) != 1:
                self._ambiguous_pair_count += 1
                continue

            candidate = candidates[0]
            key = (candidate.primary_id, candidate.duplicate_id)
            previous = self._pending.get(key)
            consecutive_frames = (
                previous.consecutive_frames + 1
                if previous is not None and previous.last_frame == frame_id - 1
                else 1
            )
            candidate.consecutive_frames = consecutive_frames
            current_pending[key] = candidate
            suppress_ids.add(candidate.duplicate_id)

            if consecutive_frames >= self.confirmation_frames:
                existing_alias = self._aliases.get(candidate.duplicate_id)
                self._aliases[candidate.duplicate_id] = _DuplicateAlias(
                    primary_id=candidate.primary_id,
                    last_confirmed_frame=frame_id,
                )
                if existing_alias is None or existing_alias.primary_id != candidate.primary_id:
                    self._confirmed_pairs.append(
                        {
                            "primary_track_id": candidate.primary_id,
                            "duplicate_track_id": candidate.duplicate_id,
                            "confirmed_frame_id": frame_id,
                            "containment_ratio": round(candidate.containment_ratio, 3),
                            "iou": round(candidate.iou, 3),
                            "bottom_distance_px": round(
                                candidate.bottom_distance_px,
                                2,
                            ),
                        }
                    )
            else:
                self._pending_duplicate_observation_count += 1

        self._pending = current_pending
        for duplicate_id, alias in list(self._aliases.items()):
            pair = (alias.primary_id, duplicate_id)
            if (
                alias.primary_id in current_ids
                and duplicate_id in current_ids
                and pair not in current_pending
            ):
                self._aliases.pop(duplicate_id, None)

        output: list[Track] = []
        for track in tracks:
            alias = self._aliases.get(track.track_id)
            if alias is not None:
                primary_visible = alias.primary_id in current_ids
                if primary_visible:
                    self._suppressed_observation_count += 1
                    continue

                if frame_id - alias.last_confirmed_frame <= self.max_handoff_gap_frames:
                    self._handoff_count += 1
                    output.append(self._with_track_id(track, alias.primary_id))
                    continue

                self._aliases.pop(track.track_id, None)

            if track.track_id in suppress_ids:
                self._suppressed_observation_count += 1
                continue

            output.append(track)

        return output

    def snapshot(self) -> dict[str, object]:
        return {
            "confirmed_pair_count": len(self._confirmed_pairs),
            "pending_duplicate_observation_count": (
                self._pending_duplicate_observation_count
            ),
            "suppressed_observation_count": self._suppressed_observation_count,
            "ambiguous_pair_count": self._ambiguous_pair_count,
            "handoff_count": self._handoff_count,
            "confirmed_pairs": self._confirmed_pairs,
        }

    def _candidate(
        self,
        left: Track,
        right: Track,
        frame_id: int,
    ) -> _PendingDuplicate | None:
        if left.class_id != right.class_id:
            return None

        left_state = self._states[left.track_id]
        right_state = self._states[right.track_id]
        primary, duplicate = self._select_primary(left_state, right_state)
        if primary is None or duplicate is None:
            return None
        if primary.observed_frames < 3:
            return None

        intersection_area, smaller_area, area_ratio = self._box_metrics(
            primary.track,
            duplicate.track,
        )
        if smaller_area == 0:
            return None

        containment_ratio = intersection_area / smaller_area
        iou = calculate_iou(primary.track.bbox, duplicate.track.bbox)
        bottom_distance_px = dist(
            primary.track.bottom_center,
            duplicate.track.bottom_center,
        )
        if containment_ratio < self.min_containment_ratio:
            return None
        if bottom_distance_px > self.max_bottom_distance_px:
            return None
        if area_ratio > self.max_area_ratio:
            return None
        if not self._motion_is_compatible(primary, duplicate):
            return None

        return _PendingDuplicate(
            primary_id=primary.track.track_id,
            duplicate_id=duplicate.track.track_id,
            first_frame=frame_id,
            last_frame=frame_id,
            consecutive_frames=1,
            containment_ratio=containment_ratio,
            iou=iou,
            bottom_distance_px=bottom_distance_px,
        )

    @staticmethod
    def _select_primary(
        left: _ActiveState,
        right: _ActiveState,
    ) -> tuple[_ActiveState | None, _ActiveState | None]:
        if left.first_seen_frame < right.first_seen_frame:
            return left, right
        if right.first_seen_frame < left.first_seen_frame:
            return right, left
        return None, None

    def _motion_is_compatible(
        self,
        primary: _ActiveState,
        duplicate: _ActiveState,
    ) -> bool:
        if primary.previous_track is None or duplicate.previous_track is None:
            return True

        primary_motion = self._motion(primary)
        duplicate_motion = self._motion(duplicate)
        return dist(primary_motion, duplicate_motion) <= self.max_motion_difference_px

    @staticmethod
    def _motion(state: _ActiveState) -> tuple[float, float]:
        if state.previous_track is None or state.previous_seen_frame is None:
            return (0.0, 0.0)
        elapsed_frames = max(1, state.last_seen_frame - state.previous_seen_frame)
        return (
            (
                state.track.bottom_center[0]
                - state.previous_track.bottom_center[0]
            )
            / elapsed_frames,
            (
                state.track.bottom_center[1]
                - state.previous_track.bottom_center[1]
            )
            / elapsed_frames,
        )

    def _observe(self, track: Track, frame_id: int) -> None:
        state = self._states.get(track.track_id)
        if state is None:
            self._states[track.track_id] = _ActiveState(
                track=track,
                previous_track=None,
                first_seen_frame=frame_id,
                last_seen_frame=frame_id,
                previous_seen_frame=None,
                observed_frames=1,
            )
            return

        state.previous_track = state.track
        state.previous_seen_frame = state.last_seen_frame
        state.track = track
        state.last_seen_frame = frame_id
        state.observed_frames += 1

    def _cleanup(self, frame_id: int) -> None:
        stale_ids = [
            track_id
            for track_id, state in self._states.items()
            if frame_id - state.last_seen_frame > self.state_retention_frames
        ]
        for track_id in stale_ids:
            self._states.pop(track_id, None)

        self._pending = {
            key: pending
            for key, pending in self._pending.items()
            if frame_id - pending.last_frame <= self.state_retention_frames
        }
        self._aliases = {
            duplicate_id: alias
            for duplicate_id, alias in self._aliases.items()
            if frame_id - alias.last_confirmed_frame <= self.state_retention_frames
        }

    @staticmethod
    def _box_metrics(
        left: Track,
        right: Track,
    ) -> tuple[float, float, float]:
        left_x1, left_y1, left_x2, left_y2 = left.bbox
        right_x1, right_y1, right_x2, right_y2 = right.bbox
        intersection_width = max(0, min(left_x2, right_x2) - max(left_x1, right_x1))
        intersection_height = max(0, min(left_y2, right_y2) - max(left_y1, right_y1))
        intersection_area = float(intersection_width * intersection_height)
        left_area = float(max(0, left_x2 - left_x1) * max(0, left_y2 - left_y1))
        right_area = float(max(0, right_x2 - right_x1) * max(0, right_y2 - right_y1))
        smaller_area = min(left_area, right_area)
        area_ratio = (
            max(left_area, right_area) / smaller_area
            if smaller_area > 0
            else float("inf")
        )
        return intersection_area, smaller_area, area_ratio

    @staticmethod
    def _with_track_id(track: Track, track_id: int) -> Track:
        return Track(
            track_id=track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            class_id=track.class_id,
            class_name=track.class_name,
            centroid=track.centroid,
            bottom_center=track.bottom_center,
        )
