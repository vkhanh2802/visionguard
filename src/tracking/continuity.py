from dataclasses import dataclass
from math import dist

from src.tracking.types import Track


Point = tuple[int, int]


@dataclass
class _ContinuityState:
    canonical_id: int
    last_track: Track
    previous_track: Track | None
    last_seen_frame: int
    previous_seen_frame: int | None
    observed_frames: int


@dataclass
class _PendingMatch:
    canonical_id: int
    last_track: Track
    last_seen_frame: int
    confirmations: int
    source_track: Track
    source_frame: int


class TrackContinuityManager:
    """Conservatively map replacement tracker IDs to canonical IDs.

    A replacement ID is hidden until it is observed for the configured number of
    consecutive frames. This prevents event engines from creating a new visit on
    the first ambiguous replacement frame.
    """

    def __init__(
        self,
        max_gap_frames: int = 15,
        min_gap_frames: int = 2,
        max_distance_px: float = 60.0,
        confirmation_frames: int = 2,
        ambiguity_margin_px: float = 15.0,
        max_size_ratio: float = 2.0,
    ):
        if max_gap_frames < 0:
            raise ValueError("max_gap_frames must be non-negative.")
        if min_gap_frames < 1:
            raise ValueError("min_gap_frames must be at least 1.")
        if min_gap_frames > max_gap_frames:
            raise ValueError("min_gap_frames cannot exceed max_gap_frames.")
        if max_distance_px <= 0:
            raise ValueError("max_distance_px must be positive.")
        if confirmation_frames < 1:
            raise ValueError("confirmation_frames must be at least 1.")
        if ambiguity_margin_px < 0:
            raise ValueError("ambiguity_margin_px must be non-negative.")
        if max_size_ratio < 1.0:
            raise ValueError("max_size_ratio must be at least 1.")

        self.max_gap_frames = max_gap_frames
        self.min_gap_frames = min_gap_frames
        self.max_distance_px = max_distance_px
        self.confirmation_frames = confirmation_frames
        self.ambiguity_margin_px = ambiguity_margin_px
        self.max_size_ratio = max_size_ratio

        self._states: dict[int, _ContinuityState] = {}
        self._used_canonical_ids: set[int] = set()
        self._raw_to_canonical: dict[int, int] = {}
        self._pending: dict[int, _PendingMatch] = {}
        self._replacement_match_count = 0
        self._pending_match_count = 0
        self._rejected_ambiguous_match_count = 0
        self._rejected_gap_match_count = 0
        self._replacement_matches: list[dict[str, int | float]] = []

    def process(self, tracks: list[Track], frame_id: int) -> list[Track]:
        self._cleanup(frame_id)
        output: list[Track] = []
        assigned_canonical_ids: set[int] = set()
        assigned_raw_ids: set[int] = set()
        new_tracks: list[Track] = []

        for track in tracks:
            canonical_id = self._raw_to_canonical.get(track.track_id)
            state = self._states.get(canonical_id) if canonical_id is not None else None

            if state is None or canonical_id in assigned_canonical_ids:
                new_tracks.append(track)
                continue

            self._update_state(state, track, frame_id)
            assigned_canonical_ids.add(canonical_id)
            assigned_raw_ids.add(track.track_id)
            output.append(self._with_track_id(track, canonical_id))

        reserved_canonical_ids = {
            pending.canonical_id
            for pending in self._pending.values()
            if pending.canonical_id not in assigned_canonical_ids
        }
        remaining_tracks: list[Track] = []

        for track in new_tracks:
            pending = self._pending.get(track.track_id)
            if pending is None or pending.canonical_id in assigned_canonical_ids:
                remaining_tracks.append(track)
                continue

            state = self._states.get(pending.canonical_id)
            if state is None or not self._is_compatible(
                pending.last_track,
                track,
                frame_id - pending.last_seen_frame,
            ):
                self._pending.pop(track.track_id, None)
                remaining_tracks.append(track)
                continue

            pending.last_track = track
            pending.last_seen_frame = frame_id
            pending.confirmations += 1
            reserved_canonical_ids.add(pending.canonical_id)

            if pending.confirmations < self.confirmation_frames:
                self._pending_match_count += 1
                continue

            canonical_id = pending.canonical_id
            self._pending.pop(track.track_id, None)
            self._raw_to_canonical[track.track_id] = canonical_id
            self._update_state(state, track, frame_id)
            assigned_canonical_ids.add(canonical_id)
            assigned_raw_ids.add(track.track_id)
            output.append(self._with_track_id(track, canonical_id))
            self._replacement_match_count += 1
            self._record_replacement_match(
                canonical_id=canonical_id,
                raw_id=track.track_id,
                frame_id=frame_id,
                previous_frame=pending.source_frame,
                previous_track=pending.source_track,
                current_track=track,
            )

        candidate_tracks = [
            track for track in remaining_tracks if track.track_id not in assigned_raw_ids
        ]
        candidate_states = [
            state
            for state in self._states.values()
            if state.canonical_id not in assigned_canonical_ids
            and state.canonical_id not in reserved_canonical_ids
            and frame_id > state.last_seen_frame
        ]

        matches = self._find_matches(candidate_tracks, candidate_states, frame_id)
        matched_raw_ids: set[int] = set()

        for track, state in matches:
            matched_raw_ids.add(track.track_id)
            self._pending[track.track_id] = _PendingMatch(
                canonical_id=state.canonical_id,
                last_track=track,
                last_seen_frame=frame_id,
                confirmations=1,
                source_track=state.last_track,
                source_frame=state.last_seen_frame,
            )

            if self.confirmation_frames == 1:
                self._pending.pop(track.track_id)
                self._raw_to_canonical[track.track_id] = state.canonical_id
                previous_state_track = state.last_track
                previous_state_frame = state.last_seen_frame
                self._update_state(state, track, frame_id)
                assigned_canonical_ids.add(state.canonical_id)
                output.append(self._with_track_id(track, state.canonical_id))
                self._replacement_match_count += 1
                self._record_replacement_match(
                    canonical_id=state.canonical_id,
                    raw_id=track.track_id,
                    frame_id=frame_id,
                    previous_frame=previous_state_frame,
                    previous_track=previous_state_track,
                    current_track=track,
                )
            else:
                self._pending_match_count += 1

        for track in candidate_tracks:
            if track.track_id in matched_raw_ids:
                continue

            canonical_id = track.track_id
            if canonical_id in self._used_canonical_ids:
                canonical_id = self._next_canonical_id()

            state = _ContinuityState(
                canonical_id=canonical_id,
                last_track=track,
                previous_track=None,
                last_seen_frame=frame_id,
                previous_seen_frame=None,
                observed_frames=1,
            )
            self._states[canonical_id] = state
            self._used_canonical_ids.add(canonical_id)
            self._raw_to_canonical[track.track_id] = canonical_id
            assigned_canonical_ids.add(canonical_id)
            output.append(self._with_track_id(track, canonical_id))

        return output

    def snapshot(self) -> dict[str, int]:
        return {
            "replacement_match_count": self._replacement_match_count,
            "pending_match_count": self._pending_match_count,
            "rejected_ambiguous_match_count": self._rejected_ambiguous_match_count,
            "rejected_gap_match_count": self._rejected_gap_match_count,
            "replacement_matches": self._replacement_matches,
        }

    def _record_replacement_match(
        self,
        canonical_id: int,
        raw_id: int,
        frame_id: int,
        previous_frame: int,
        previous_track: Track,
        current_track: Track,
    ) -> None:
        self._replacement_matches.append(
            {
                "canonical_track_id": canonical_id,
                "replacement_track_id": raw_id,
                "frame_id": frame_id,
                "gap_frames": frame_id - previous_frame,
                "distance_px": round(
                    dist(previous_track.bottom_center, current_track.bottom_center),
                    2,
                ),
            }
        )

    def _find_matches(
        self,
        tracks: list[Track],
        states: list[_ContinuityState],
        frame_id: int,
    ) -> list[tuple[Track, _ContinuityState]]:
        candidates: list[tuple[float, Track, _ContinuityState]] = []
        distances_by_track: dict[int, list[float]] = {}

        for track in tracks:
            for state in states:
                gap_frames = frame_id - state.last_seen_frame
                if not self._is_compatible(
                    state.last_track,
                    track,
                    gap_frames,
                ):
                    if gap_frames > self.max_gap_frames:
                        self._rejected_gap_match_count += 1
                    continue

                if state.observed_frames < 3:
                    continue
                if gap_frames < self.min_gap_frames:
                    continue

                distance = self._predicted_distance(state, track, gap_frames)
                if distance > self.max_distance_px:
                    continue

                candidates.append((distance, track, state))
                distances_by_track.setdefault(track.track_id, []).append(distance)

        ambiguous_tracks = {
            track_id
            for track_id, distances in distances_by_track.items()
            if len(distances) > 1
            and sorted(distances)[1] - sorted(distances)[0]
            <= self.ambiguity_margin_px
        }
        self._rejected_ambiguous_match_count += len(ambiguous_tracks)

        matches: list[tuple[Track, _ContinuityState]] = []
        assigned_tracks: set[int] = set()
        assigned_states: set[int] = set()

        for distance, track, state in sorted(candidates, key=lambda item: item[0]):
            if track.track_id in ambiguous_tracks:
                continue
            if track.track_id in assigned_tracks or state.canonical_id in assigned_states:
                continue

            assigned_tracks.add(track.track_id)
            assigned_states.add(state.canonical_id)
            matches.append((track, state))

        return matches

    def _predicted_distance(
        self,
        state: _ContinuityState,
        track: Track,
        gap_frames: int,
    ) -> float:
        last_position = state.last_track.bottom_center
        predicted_position = last_position

        if state.previous_track is not None and state.previous_seen_frame is not None:
            elapsed = state.last_seen_frame - state.previous_seen_frame
            if elapsed > 0:
                velocity = (
                    (last_position[0] - state.previous_track.bottom_center[0]) / elapsed,
                    (last_position[1] - state.previous_track.bottom_center[1]) / elapsed,
                )
                predicted_position = (
                    round(last_position[0] + velocity[0] * gap_frames),
                    round(last_position[1] + velocity[1] * gap_frames),
                )

        return dist(predicted_position, track.bottom_center)

    def _is_compatible(
        self,
        previous_track: Track,
        track: Track,
        gap_frames: int,
    ) -> bool:
        if previous_track.class_id != track.class_id:
            return False
        if gap_frames > self.max_gap_frames:
            return False
        if dist(previous_track.bottom_center, track.bottom_center) > self.max_distance_px:
            return False

        previous_width = max(1, previous_track.bbox[2] - previous_track.bbox[0])
        previous_height = max(1, previous_track.bbox[3] - previous_track.bbox[1])
        current_width = max(1, track.bbox[2] - track.bbox[0])
        current_height = max(1, track.bbox[3] - track.bbox[1])
        width_ratio = max(previous_width, current_width) / min(
            previous_width,
            current_width,
        )
        height_ratio = max(previous_height, current_height) / min(
            previous_height,
            current_height,
        )
        return max(width_ratio, height_ratio) <= self.max_size_ratio

    def _update_state(
        self,
        state: _ContinuityState,
        track: Track,
        frame_id: int,
    ) -> None:
        state.previous_track = state.last_track
        state.previous_seen_frame = state.last_seen_frame
        state.last_track = track
        state.last_seen_frame = frame_id
        state.observed_frames += 1

    def _cleanup(self, frame_id: int) -> None:
        stale_ids = [
            canonical_id
            for canonical_id, state in self._states.items()
            if frame_id - state.last_seen_frame > self.max_gap_frames
        ]
        self._rejected_gap_match_count += len(stale_ids)
        for canonical_id in stale_ids:
            self._states.pop(canonical_id, None)

        self._raw_to_canonical = {
            raw_id: canonical_id
            for raw_id, canonical_id in self._raw_to_canonical.items()
            if canonical_id in self._states
        }
        self._pending = {
            raw_id: pending
            for raw_id, pending in self._pending.items()
            if (
                pending.canonical_id in self._states
                and frame_id - pending.last_seen_frame <= self.max_gap_frames
            )
        }

    def _next_canonical_id(self) -> int:
        canonical_id = max(self._used_canonical_ids, default=0) + 1
        while canonical_id in self._used_canonical_ids:
            canonical_id += 1
        return canonical_id

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
