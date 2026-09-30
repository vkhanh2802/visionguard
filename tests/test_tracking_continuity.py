from src.tracking import Track, TrackContinuityManager


def make_track(
    track_id: int,
    point: tuple[int, int],
    size: tuple[int, int] = (20, 40),
    class_id: int = 0,
) -> Track:
    x, y = point
    width, height = size
    return Track(
        track_id=track_id,
        bbox=(x - width // 2, y - height, x + width // 2, y),
        confidence=0.9,
        class_id=class_id,
        class_name="person" if class_id == 0 else "other",
        centroid=(x, y - height // 2),
        bottom_center=point,
    )


def test_confirms_replacement_id_before_exposing_canonical_track():
    manager = TrackContinuityManager(
        max_gap_frames=5,
        min_gap_frames=2,
        max_distance_px=20,
        confirmation_frames=2,
    )

    assert [track.track_id for track in manager.process([make_track(1, (50, 50))], 1)] == [1]
    manager.process([make_track(1, (50, 50))], 2)
    manager.process([make_track(1, (50, 50))], 3)
    assert manager.process([], 4) == []

    # The first replacement observation is held out of event processing.
    assert manager.process([make_track(2, (53, 50))], 5) == []
    replacement = manager.process([make_track(2, (56, 50))], 6)

    assert [track.track_id for track in replacement] == [1]
    assert manager.snapshot()["replacement_match_count"] == 1


def test_does_not_reassociate_when_original_track_is_visible():
    manager = TrackContinuityManager(
        max_gap_frames=5,
        min_gap_frames=2,
        max_distance_px=20,
        confirmation_frames=1,
    )

    manager.process([make_track(1, (50, 50))], 1)
    tracks = manager.process(
        [make_track(1, (51, 50)), make_track(2, (53, 50))],
        2,
    )

    assert {track.track_id for track in tracks} == {1, 2}
    assert manager.snapshot()["replacement_match_count"] == 0


def test_rejects_ambiguous_nearby_replacement():
    manager = TrackContinuityManager(
        max_gap_frames=5,
        min_gap_frames=2,
        max_distance_px=30,
        confirmation_frames=1,
        ambiguity_margin_px=15,
    )

    manager.process(
        [make_track(1, (40, 50)), make_track(2, (60, 50))],
        1,
    )
    manager.process(
        [make_track(1, (40, 50)), make_track(2, (60, 50))],
        2,
    )
    manager.process(
        [make_track(1, (40, 50)), make_track(2, (60, 50))],
        3,
    )
    manager.process([], 4)
    tracks = manager.process([make_track(3, (50, 50))], 5)

    assert [track.track_id for track in tracks] == [3]
    assert manager.snapshot()["replacement_match_count"] == 0
    assert manager.snapshot()["rejected_ambiguous_match_count"] == 1


def test_matches_nearby_replacements_one_to_one():
    manager = TrackContinuityManager(
        max_gap_frames=5,
        min_gap_frames=2,
        max_distance_px=15,
        confirmation_frames=1,
        ambiguity_margin_px=2,
    )

    manager.process(
        [make_track(1, (30, 50)), make_track(2, (70, 50))],
        1,
    )
    manager.process(
        [make_track(1, (30, 50)), make_track(2, (70, 50))],
        2,
    )
    manager.process(
        [make_track(1, (30, 50)), make_track(2, (70, 50))],
        3,
    )
    manager.process([], 4)
    tracks = manager.process(
        [make_track(3, (31, 50)), make_track(4, (69, 50))],
        5,
    )

    assert {track.track_id for track in tracks} == {1, 2}
    assert manager.snapshot()["replacement_match_count"] == 2


def test_rejects_replacement_after_maximum_gap():
    manager = TrackContinuityManager(
        max_gap_frames=2,
        min_gap_frames=2,
        max_distance_px=20,
        confirmation_frames=1,
    )

    manager.process([make_track(1, (50, 50))], 1)
    manager.process([make_track(1, (50, 50))], 2)
    manager.process([make_track(1, (50, 50))], 3)
    tracks = manager.process([make_track(2, (52, 50))], 6)

    assert [track.track_id for track in tracks] == [2]
    assert manager.snapshot()["replacement_match_count"] == 0
    assert manager.snapshot()["rejected_gap_match_count"] == 1


def test_rejects_replacement_with_incompatible_box_size():
    manager = TrackContinuityManager(
        max_gap_frames=5,
        min_gap_frames=2,
        max_distance_px=20,
        confirmation_frames=1,
        max_size_ratio=1.5,
    )

    manager.process([make_track(1, (50, 50), size=(20, 40))], 1)
    manager.process([make_track(1, (50, 50), size=(20, 40))], 2)
    manager.process([make_track(1, (50, 50), size=(20, 40))], 3)
    manager.process([], 4)
    tracks = manager.process([make_track(2, (52, 50), size=(40, 80))], 5)

    assert [track.track_id for track in tracks] == [2]
    assert manager.snapshot()["replacement_match_count"] == 0
