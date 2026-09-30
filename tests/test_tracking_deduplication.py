from src.tracking import ActiveTrackDeduplicator, Track


def make_track(
    track_id: int,
    point: tuple[int, int],
    size: tuple[int, int] = (30, 60),
) -> Track:
    x, y = point
    width, height = size
    return Track(
        track_id=track_id,
        bbox=(x - width // 2, y - height, x + width // 2, y),
        confidence=0.9,
        class_id=0,
        class_name="person",
        centroid=(x, y - height // 2),
        bottom_center=point,
    )


def test_suppresses_confirmed_contained_duplicate_box():
    deduplicator = ActiveTrackDeduplicator(confirmation_frames=3)

    for frame_id in range(1, 4):
        assert len(deduplicator.process([make_track(1, (50, 50))], frame_id)) == 1

    duplicate = make_track(2, (51, 50), size=(20, 40))
    assert [track.track_id for track in deduplicator.process(
        [make_track(1, (50, 50)), duplicate],
        4,
    )] == [1]
    assert [track.track_id for track in deduplicator.process(
        [make_track(1, (50, 50)), duplicate],
        5,
    )] == [1]
    assert [track.track_id for track in deduplicator.process(
        [make_track(1, (50, 50)), duplicate],
        6,
    )] == [1]

    snapshot = deduplicator.snapshot()
    assert snapshot["confirmed_pair_count"] == 1
    assert snapshot["suppressed_observation_count"] == 3


def test_retains_state_without_enabling_handoff():
    deduplicator = ActiveTrackDeduplicator(
        confirmation_frames=3,
        state_retention_frames=5,
        max_handoff_gap_frames=0,
    )

    for frame_id in range(1, 4):
        deduplicator.process([make_track(1, (50, 50))], frame_id)

    deduplicator.process([], 4)
    duplicate = make_track(2, (51, 50), size=(20, 40))
    for frame_id in (5, 6):
        tracks = deduplicator.process(
            [make_track(1, (50, 50)), duplicate],
            frame_id,
        )
        assert [track.track_id for track in tracks] == [1]

    confirmed = deduplicator.process(
        [make_track(1, (50, 50)), duplicate],
        7,
    )
    assert [track.track_id for track in confirmed] == [1]

    no_handoff = deduplicator.process([duplicate], 8)
    assert [track.track_id for track in no_handoff] == [2]
    assert deduplicator.snapshot()["confirmed_pair_count"] == 1
    assert deduplicator.snapshot()["handoff_count"] == 0


def test_releases_pending_candidate_when_overlap_breaks():
    deduplicator = ActiveTrackDeduplicator(confirmation_frames=3)

    for frame_id in range(1, 4):
        deduplicator.process([make_track(1, (50, 50))], frame_id)

    pending = deduplicator.process(
        [make_track(1, (50, 50)), make_track(2, (51, 50), size=(20, 40))],
        4,
    )
    assert [track.track_id for track in pending] == [1]

    released = deduplicator.process(
        [make_track(1, (50, 50)), make_track(2, (100, 50), size=(20, 40))],
        5,
    )
    assert [track.track_id for track in released] == [1, 2]
    assert deduplicator.snapshot()["confirmed_pair_count"] == 0


def test_requarantines_duplicate_after_short_tracking_gap():
    deduplicator = ActiveTrackDeduplicator(
        confirmation_frames=3,
        state_retention_frames=5,
    )

    for frame_id in range(1, 4):
        deduplicator.process([make_track(1, (50, 50))], frame_id)

    duplicate = make_track(2, (51, 50), size=(20, 40))
    assert [track.track_id for track in deduplicator.process(
        [make_track(1, (50, 50)), duplicate],
        4,
    )] == [1]
    deduplicator.process([make_track(1, (50, 50))], 5)

    reacquired = deduplicator.process(
        [make_track(1, (50, 50)), make_track(2, (52, 50), size=(20, 40))],
        6,
    )

    assert [track.track_id for track in reacquired] == [1]


def test_handoffs_confirmed_duplicate_when_primary_disappears():
    deduplicator = ActiveTrackDeduplicator(confirmation_frames=1)

    for frame_id in range(1, 4):
        deduplicator.process([make_track(1, (50, 50))], frame_id)

    deduplicator.process(
        [make_track(1, (50, 50)), make_track(2, (51, 50), size=(20, 40))],
        4,
    )
    handoff = deduplicator.process(
        [make_track(2, (52, 50), size=(20, 40))],
        5,
    )

    assert [track.track_id for track in handoff] == [1]
    assert deduplicator.snapshot()["handoff_count"] == 1


def test_keeps_nearby_people_when_boxes_are_not_contained():
    deduplicator = ActiveTrackDeduplicator(confirmation_frames=1)

    for frame_id in range(1, 4):
        deduplicator.process(
            [make_track(1, (40, 50)), make_track(2, (80, 50))],
            frame_id,
        )

    tracks = deduplicator.process(
        [make_track(1, (42, 50)), make_track(2, (78, 50))],
        4,
    )

    assert {track.track_id for track in tracks} == {1, 2}
    assert deduplicator.snapshot()["confirmed_pair_count"] == 0


def test_rejects_ambiguous_duplicate_candidate():
    deduplicator = ActiveTrackDeduplicator(confirmation_frames=1)

    deduplicator.process([make_track(1, (45, 50))], 1)
    for frame_id in (2, 3):
        deduplicator.process(
            [make_track(1, (45, 50)), make_track(2, (55, 50))],
            frame_id,
        )

    tracks = deduplicator.process(
        [
            make_track(1, (45, 50)),
            make_track(2, (55, 50)),
            make_track(3, (50, 50), size=(20, 40)),
        ],
        4,
    )

    assert {track.track_id for track in tracks} == {1, 2, 3}
    assert deduplicator.snapshot()["ambiguous_pair_count"] == 1
