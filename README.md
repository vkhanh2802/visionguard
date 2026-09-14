# VisionGuard

VisionGuard is a real-time video analytics project for object detection, multi-object tracking, and event understanding.

## Current Status

Week 1 completed:

- OpenCV video pipeline
- YOLO26 object detection
- Confidence filtering
- Class filtering
- Bounding-box visualization
- Processing FPS measurement
- Video output
- CLI interface

Week 2 completed:

- ByteTrack integration
- Persistent object IDs
- Track abstraction
- Centroid and bottom-center extraction
- Track trajectory visualization
- Track history cleanup
- IoU implementation
- Geometry unit tests

Week 3 implemented and evaluated: line crossing and people counting.

Week 4 implemented and manually evaluated: restricted-zone intrusion and loitering.
The current baseline includes three independent event engines, annotated video output,
and a three-video evaluation. See [Week 4 evaluation](docs/week4_evaluation.md)
for results, failure analysis, and remaining validation items.

## Pipeline

```text
Video
  |
  v
YOLO
  |
  v
ByteTrack
  |
  v
Track[]
  |-- LineCrossingEngine
  |-- IntrusionEngine
  |-- LoiteringEngine
  |
  v
Event[]
  |
  v
Counters + Console Logging + Visualization
  |
  v
Annotated Output Video
```

## Installation

```bash
conda env create -f environment.yml
conda activate visionguard
```

## Usage

```bash
python -m scripts.run_video --source data/videos/test.mp4
```

## Example

```bash
python -m scripts.run_video --source data/videos/test.mp4 --conf 0.4 --output data/outputs/output.mp4
```

## Tracking Confidence Experiment

To evaluate the effect of detection confidence on tracking stability, the same crowded video was tested with different confidence thresholds.

### Observation

With `conf=0.2`, the tracker generated new track IDs much more frequently. When a person was temporarily missed because of occlusion or weak detection, the person often reappeared with a new and significantly larger track ID. The overall track ID count also increased very quickly, indicating a larger number of short-lived or noisy tracks.

With `conf=0.4`, track IDs still increased over time, which is expected when new objects enter the scene, but the increase was noticeably slower. More importantly, some people who were temporarily missed were successfully associated with their previous track ID when they reappeared. This behavior was less stable with `conf=0.2`.

### Comparison

| Confidence | Tracking behavior |
| ---------- | ----------------- |
| `0.2` | More weak detections, rapid growth in track IDs, more short-lived tracks, and more frequent ID reassignment after temporary misses |
| `0.4` | Fewer noisy tracks, slower ID growth, and better ID continuity for several objects after short occlusions |

### Conclusion

For the tested crowded scene, lowering the confidence threshold did not improve tracking quality. Although a lower threshold provides more detections to ByteTrack, it also introduces additional low-confidence and noisy detections that can interfere with track association and create unnecessary new tracks.

In this experiment, `conf=0.4` provided a better balance between detection coverage and tracking stability. It produced fewer fragmented tracks and showed better identity continuity after temporary missed detections.

Therefore, `0.4` is currently used as the default detection confidence threshold for VisionGuard. This value is treated as a scene-dependent hyperparameter rather than a universal optimal threshold and may be adjusted for different camera conditions or datasets.

## Week 3 - Line Crossing & People Counting

Implemented:

- Virtual line crossing detection
- IN / OUT direction classification
- Finite line-segment intersection
- Pixel-based dead zone
- Bottom-center based crossing geometry
- Stateful crossing detection
- Multi-frame side confirmation
- Track state cleanup
- IN / OUT / NET counters
- Unit tests for geometry and event state

## Line Crossing Evaluation

The line-crossing system was evaluated on three videos with increasing levels of difficulty:

| Video | Scenario | Ground Truth IN | Ground Truth OUT |
| ----- | -------- | --------------: | ---------------: |
| A | Simple scene, few people, low occlusion | 2 | 4 |
| B | Medium scene, more people, turn-backs and near-line movement | 8 | 5 |
| C | Crowded scene with occlusion and unstable tracking | 2 | 12 |

### Dead-Zone Experiment

A pixel-based dead zone was evaluated to reduce false crossing events caused by bounding-box jitter around the virtual line.

#### Video A

| Dead Zone | Predicted IN | Predicted OUT | False Events | Missed Events |
| --------: | -----------: | ------------: | -----------: | ------------: |
| 3 px | 2 | 3 | 0 | 1 |
| 5 px | 2 | 3 | 0 | 1 |
| 8 px | 2 | 3 | 0 | 1 |
| 10 px | 2 | 3 | 0 | 1 |

The missed event was caused by two people moving very close together, resulting in the detector producing one bounding box instead of two separate detections.

#### Video B

| Dead Zone | Predicted IN | Predicted OUT | False Events | Missed Events |
| --------: | -----------: | ------------: | -----------: | ------------: |
| 3 px | 7 | 4 | 0 | 2 |
| 5 px | 7 | 4 | 0 | 2 |
| 8 px | 6 | 4 | 0 | 3 |
| 10 px | 5 | 4 | 0 | 4 |

One missed event was caused by overlapping bounding boxes. Other missed events were observed when tracking became unstable or temporarily disappeared near the virtual line.

Increasing the dead zone beyond 5 pixels also increased the number of missed crossings.

#### Video C

| Dead Zone | Predicted IN | Predicted OUT | False Events | Missed Events |
| --------: | -----------: | ------------: | -----------: | ------------: |
| 3 px | 5 | 13 | 8 | 4 |
| 5 px | 3 | 12 | 5 | 4 |
| 8 px | 2 | 11 | 3 | 4 |
| 10 px | 2 | 11 | 3 | 4 |

The crowded scene produced significantly more false crossing events. These were mainly caused by bounding-box positions oscillating around the virtual line.

Increasing the dead zone reduced false events from 8 at 3 pixels to 3 at 8 pixels.

The remaining missed events were mainly caused by overlapping bounding boxes and tracking instability under heavy occlusion.

### Failure Analysis

| Failure | Observed Cause | Pipeline Layer |
| ------- | -------------- | -------------- |
| Two people counted as one | Nearby people merged into one bounding box | Detection |
| Missed crossing | Overlapping bounding boxes | Detection |
| Missed crossing near the line | Track temporarily lost or fragmented | Tracking |
| Duplicate / false crossing | Bounding-box jitter around the virtual line | Tracking / Event |
| New identity after occlusion | ByteTrack assigns a new track ID | Tracking |

### Conclusion

The experiments show that the line-crossing dead zone provides a trade-off between false-event suppression and crossing recall.

A small dead zone such as `3 px` is more sensitive to bounding-box jitter, especially in crowded scenes. Increasing the dead zone reduces false crossings, but overly large values such as `8-10 px` can cause valid crossings to be missed.

Based on the three evaluated videos, VisionGuard currently uses:

```text
Detection confidence: 0.4
Line-crossing dead zone: 5 px
```

A `5 px` dead zone was selected as the current default because it provides a practical balance between event stability and crossing recall across the tested scenes.

The experiments also show that many remaining counting errors originate upstream from detection and tracking rather than from the line-crossing state machine itself.

### Current Limitations

- Counting accuracy depends on stable object detection and tracking.
- Closely overlapping people may be merged into a single bounding box.
- Long or heavy occlusion can cause track loss or ID reassignment.
- Bounding-box jitter near the virtual line can still produce false events in crowded scenes.
- The virtual line and dead-zone threshold currently require manual configuration for each camera.
- The current geometry operates in image coordinates and does not compensate for perspective distortion.

### Side Confirmation Experiment

To further reduce false crossing events caused by bounding-box oscillation around the virtual line, a side-confirmation mechanism was evaluated on the crowded test video.

A crossing transition is accepted only when a track remains on the new side of the virtual line for a specified number of consecutive frames.

| Confirmation Frames | Predicted IN | Predicted OUT | False Events | Missed Events |
| ------------------: | -----------: | ------------: | -----------: | ------------: |
| 1 | 3 | 11 | 4 | 4 |
| 2 | 3 | 11 | 4 | 4 |
| 3 | 2 | 10 | 2 | 4 |
| 5 | 2 | 10 | 2 | 4 |

Increasing the confirmation requirement from 1 to 3 frames reduced false crossing events from 4 to 2 without increasing the number of missed crossings.

Using 5 confirmation frames did not provide any additional improvement compared with 3 frames and would introduce additional event latency.

Therefore, VisionGuard currently uses:

- Detection confidence: `0.4`
- Line dead zone: `5 px`
- Side confirmation: `3 frames`

The confirmation mechanism is particularly useful in crowded scenes where bounding-box positions can oscillate around the virtual line for several consecutive frames.

### Remaining Failure Cases

After applying a 5-pixel dead zone and 3-frame side confirmation, false crossing events were reduced significantly.

The remaining errors were mainly associated with upstream computer-vision failures:

- Overlapping people producing unstable or merged bounding boxes
- Temporary loss of tracks near the virtual line
- ID fragmentation after occlusion
- Prolonged bounding-box instability in crowded areas

These errors cannot be fully corrected by the line-crossing state machine alone because the event engine depends on the quality and identity consistency of upstream tracks.

### Final Week 3 Configuration

- Detection confidence: `0.4`
- Dead zone: `5 px`
- Side confirmation: `3 frames`
- Maximum missing frames: `30`

These values were selected empirically on three videos with different levels of crowding and occlusion.

## Week 4 - Restricted-Zone Intrusion & Loitering

Implemented:

- Polygon ROI and self-implemented ray-casting point-in-polygon geometry
- Bottom-center based ROI membership checks
- Intrusion events on observed outside-to-inside transitions
- Loitering events based on video timestamps and a configurable dwell threshold
- Independent per-track state and stale-state cleanup
- Combined line crossing, intrusion, and loitering processing on every frame
- ROI outlines, track trajectories, event counters, and console event logs
- Unit tests for polygon geometry, intrusion, loitering, and multi-engine integration

### Design Decisions

- Points on polygon edges or vertices are treated as inside the ROI.
- A track's first observation does not produce an intrusion event, even if inside.
- Loitering starts at the first observed inside position and triggers once per visit.
- An observed exit resets the loitering visit; a later re-entry can produce new events.
- Short tracking gaps retain state within the configured frame-gap limit. Loitering
  duration includes that gap, but an event is emitted only on an observed inside track.
- Dwell time measures presence in the ROI; the person does not need to stand still.
- Event engines perform no drawing; visualization is handled separately.

### Evaluation Summary

Three videos were manually evaluated with a **5-second loitering threshold**.
Intrusion results below are calculated from the author's reported event matches:

| Video | Ground Truth | Predicted | TP | FP | FN | Precision | Recall | F1 |
| ----- | -----------: | --------: | -: | -: | -: | --------: | -----: | -: |
| A | 6 | 5 | 5 | 0 | 1 | 100.00% | 83.33% | 90.91% |
| B | 10 | 12 | 10 | 2 | 0 | 83.33% | 100.00% | 90.91% |
| C | 2 | 2 | 2 | 0 | 0 | 100.00% | 100.00% | 100.00% |
| **Total (micro)** | **18** | **19** | **17** | **2** | **1** | **89.47%** | **94.44%** | **91.89%** |

Loitering detected **1/1 reported ground-truth event**, with no reported false
loitering events across the three clips. This is a small baseline with only one
positive loitering example, not evidence of general-purpose accuracy.

Observed failures:

- **Video A:** one missed intrusion when two nearby people shared a single bounding box.
- **Video B:** two false intrusions caused by bottom-center jitter across the ROI boundary.
- **Video C:** no reported event errors; part of the configured ROI extends beyond the image.

Full ROI coordinates, clip metadata, methodology, and follow-ups are documented in
[docs/week4_evaluation.md](docs/week4_evaluation.md).

### Running the Week 4 Baseline

Set `RESTRICTED_ZONE`, `LINE_START`, `LINE_END`, and
`LOITERING_THRESHOLD_SECONDS` in `scripts/run_video.py` for the selected video.
ROI coordinates are in original-frame pixels and are camera-specific. The report
contains the three evaluated ROIs; the current script is configured for video C.

```bash
python -m scripts.run_video --source path/to/video.mp4 --model yolo26n.pt --conf 0.4 --output data/outputs/week4_demo.mp4 --no-display
python -m pytest tests/ -q
```

Replace the source path with your local video. The local demo outputs are
`data/outputs/week4videoA.mp4`, `data/outputs/week4videoB.mp4`, and
`data/outputs/week4videoC.mp4`. Input/output videos are ignored by Git and are not
bundled with a clone of this repository.

### Remaining Validation and Improvements

- Confirm a full test-suite pass; no new pytest pass is claimed by this documentation update.
- Check cleanup boundary cases, including continuous observations with
  `max_missing_frames=0` and exact-limit tracking gaps.
- Add stable inside/outside confirmation to reduce boundary-jitter duplicates,
  then re-evaluate all three clips for any loss in recall.
- Add event timestamps and track references to the annotations, measure loitering
  trigger delay, and expand the number of positive loitering examples.
- ID switches can reset dwell timers or cause repeated alerts for the same physical
  person; short-gap continuity assumes no unobserved exit and re-entry.
