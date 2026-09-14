# Week 4 Evaluation - Intrusion and Loitering

## 1. Scope and Evidence

This report records the author's manual evaluation of the Week 4 baseline on
three videos. It evaluates restricted-zone intrusion and loitering; line-crossing
results are documented separately in the [README](../README.md#line-crossing-evaluation).

Evidence used:

- Author-provided ground-truth counts, predicted counts, and false/missed-event observations.
- Local annotated outputs in `data/outputs/week4videoA.mp4`,
  `data/outputs/week4videoB.mp4`, and `data/outputs/week4videoC.mp4`.
- Output-video metadata inspected with FFprobe and sampled frames reviewed across
  each clip. Sampled-frame inspection confirms the overlays are present but is not
  an independent frame-by-frame audit of every event match.

These are exploratory results on three manually selected clips, not a held-out
benchmark. Event-level timestamps, matching tolerances, and annotation files were
not included in the supplied survey.

## 2. Configuration and Clip Metadata

The survey explicitly records a **5.0-second loitering threshold** and the ROIs below.
Current script settings provide configuration context:

| Setting | Current value |
| ------- | ------------- |
| Model default | `yolo26n.pt` |
| Detector confidence default | `0.4` |
| Target class | `person` |
| Tracker | ByteTrack |
| ROI reference point | Bounding-box bottom-center |
| Zone identifier | `restricted-zone-1` |
| Loitering threshold | `5.0` seconds |
| Maximum missing frames | `30` |
| Line-crossing dead zone | `5.0` pixels |
| Line-crossing confirmation | `3` frames |

The last two settings apply to line crossing, not intrusion stabilization.
Per-run commands, model overrides, dependency versions, hardware, and the exact
code revision were not recorded in the survey. Consequently, current defaults
are not asserted to be an independently verified configuration snapshot of all
three historical runs.

| Video | Local output | Resolution | Encoded FPS | Frames | Duration |
| ----- | ------------ | ---------- | ----------: | -----: | -------: |
| A | `data/outputs/week4videoA.mp4` | 1920 x 1080 | 30 | 411 | 13.7 s |
| B | `data/outputs/week4videoB.mp4` | 960 x 540 | 30 | 1050 | 35.0 s |
| C | `data/outputs/week4videoC.mp4` | 1920 x 1080 | 25 | 910 | 36.4 s |

Encoded FPS describes playback rate, not processing throughput. No end-to-end
performance benchmark or latency percentile is reported here.

### ROI Coordinates

Coordinates use the original image reference system: x increases rightward and
y increases downward. Vertices are listed in boundary order.

```python
ROI_A = (
    (100, 800),
    (600, 300),
    (800, 300),
    (300, 800),
)

ROI_B = (
    (100, 450),
    (300, 450),
    (300, 350),
    (100, 350),
)

ROI_C = (
    (1400, 750),
    (2050, 750),
    (2350, 350),
    (1600, 350),
)
```

**Video C:** vertices at x=2050 and x=2350 exceed the 1920-pixel image width.
The displayed ROI is therefore clipped by the right image boundary. This can
represent a region extending beyond the visible scene, but events outside the
image cannot be evaluated. The coordinates are preserved here exactly as supplied.
If the ROI is changed to fit within the image, ground truth must be reassessed.

## 3. Event Semantics

- Polygon boundary points count as inside.
- Intrusion is an observed outside-to-inside transition for the same track ID.
- First observation inside initializes state without an intrusion event.
- Remaining inside does not create additional intrusion events; exiting and
  re-entering can create another valid intrusion.
- Loitering starts at the first observed inside timestamp and triggers when
  duration reaches or exceeds 5 seconds, once per visit.
- An observed exit resets the visit. Loitering means dwell time in the ROI,
  including movement within it, rather than detection of a stationary person.
- Short tracking gaps retain state within the implemented frame-gap limit.
  Retained loitering duration includes the gap; no event is emitted for an absent
  track. A later observed inside track can trigger the event.
- A new ID or expired state starts a new visit. This can delay or miss a true
  loitering event and can also re-alert on a physical person who already triggered
  under an earlier ID.

## 4. Ground Truth and Reported Predictions

| Video | GT Intrusion | GT Loitering | Predicted Intrusion | Predicted Loitering | False Intrusion | Missed Intrusion |
| ----- | -----------: | -----------: | ------------------: | ------------------: | --------------: | ---------------: |
| A | 6 | 0 | 5 | 0 | 0 | 1 |
| B | 10 | 1 | 12 | 1 | 2 | 0 |
| C | 2 | 0 | 2 | 0 | 0 | 0 |
| **Total** | **18** | **1** | **19** | **1** | **2** | **1** |

The supplied false/missed-event descriptions identify intrusion errors. No false
or missed loitering events were reported. Equal predicted and ground-truth counts
alone do not prove correctness; the metrics below rely on the author's manual
matching observations, including the reported correct loitering event in B.

## 5. Metrics

```text
TP = correctly matched events
FP = unmatched predictions, including duplicate alerts
FN = ground-truth events without a matching prediction

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * TP / (2 * TP + FP + FN)
```

### Intrusion

| Video | TP | FP | FN | Precision | Recall | F1 |
| ----- | -: | -: | -: | --------: | -----: | -: |
| A | 5 | 0 | 1 | 100.00% | 83.33% | 90.91% |
| B | 10 | 2 | 0 | 83.33% | 100.00% | 90.91% |
| C | 2 | 0 | 0 | 100.00% | 100.00% | 100.00% |
| **Micro aggregate** | **17** | **2** | **1** | **89.47%** | **94.44%** | **91.89%** |

The aggregate pools TP/FP/FN across videos; it is not the arithmetic mean of
per-video percentages.

### Loitering

| Video | TP | FP | FN | Precision | Recall | F1 |
| ----- | -: | -: | -: | --------: | -----: | -: |
| A | 0 | 0 | 0 | N/A | N/A | N/A |
| B | 1 | 0 | 0 | 100.00% | 100.00% | 100.00% |
| C | 0 | 0 | 0 | N/A | N/A | N/A |
| **Micro aggregate** | **1** | **0** | **0** | **100.00%** | **100.00%** | **100.00%** |

N/A indicates a zero denominator. Videos without any ground-truth or predicted
loitering events are not assigned perfect per-video scores.

The practical conclusion is **1/1 positive event detected, with no reported false
loitering alerts on these clips**. More positive examples are required before
making broader claims. Trigger delay was not measured because event timestamps
were not supplied.

## 6. Failure Analysis

### Video A - Merged Detections

The author observed two nearby people represented by one bounding box, leading
to one missed intrusion. This is an upstream detection/tracking limitation:
an event engine receiving only one track cannot recover the second person's visit.

Follow-up: preserve a timestamp and screenshot of the failure, then compare a
larger model or inference image size on the same clip. Measure the accuracy and
runtime trade-off rather than assuming a model change resolves the issue.

### Video B - Boundary Jitter

Two false intrusions were attributed to the bounding-box bottom-center oscillating
across the ROI boundary. The resulting observed sequence can be:

```text
inside -> outside (jitter) -> inside -> duplicate intrusion
```

The intrusion engine currently reacts immediately to this transition. Line-crossing
dead-zone and confirmation settings do not stabilize polygon membership.

Priority improvement: confirm both entry and exit before changing the stable ROI
state. Retain legitimate re-entry behavior; a permanent one-alert-per-ID rule would
hide real subsequent visits. Re-evaluate all three clips after any change because
confirmation may suppress brief valid entries and delay alerts. Boundary jitter
can also reset a loitering timer, although that failure was not reported here.

### Video C - Correct Counts in a Limited Visible ROI

The reported counts match both ground-truth intrusion events, with no loitering
events. This is a successful result for the observed portion of the configured
ROI, not evidence that the detector is robust to all crowding or occlusion cases.
The partly out-of-frame ROI limits what can be observed and annotated.

## 7. Reproduction and Validation

1. Obtain the original local source clip. Videos are ignored by Git and are not
   distributed with this repository.
2. Set `RESTRICTED_ZONE` in `scripts/run_video.py` to the corresponding ROI above.
   Keep the source coordinate system unchanged, or transform the ROI if resizing.
3. Set `LOITERING_THRESHOLD_SECONDS = 5.0` and record all other configuration values.
4. Run the pipeline, replacing the input placeholder and output video letter:

```bash
python -m scripts.run_video --source path/to/video_A.mp4 --model yolo26n.pt --conf 0.4 --output data/outputs/week4videoA.mp4 --no-display
python -m pytest tests/ -q
```

For future evaluation, record one annotation per physical-person visit, with
entry/exit times and the expected loitering trigger time. Match predictions
one-to-one to those annotations using a documented timing tolerance. Extra alerts
for an already matched event are FP; unmatched annotations are FN. Track IDs are
useful references but should not replace physical-person identity in ground truth.

```text
Expected loitering trigger = annotated visit start + threshold
Signed trigger delay      = predicted trigger time - expected trigger time
```

Record the exact commit, command, dependency versions, device, and model for each
rerun. The current survey lacks those run artifacts, so exact reproduction of the
historical results is not guaranteed.

Tests exist for geometry, line crossing, intrusion, loitering, and integration.
This documentation update does not certify a new full-suite pass. Remaining
cleanup validation should explicitly cover continuous observations with
`max_missing_frames=0`, a return after exactly N missing frames, and expiry after
N+1 missing frames; elapsed frame distance and actual missing-frame count differ
when a track returns.

## 8. Conclusion and Next Steps

The Week 4 baseline integrates polygon intrusion and dwell-time loitering alongside
line crossing, with visual output and manual evaluation on three videos.
Intrusion achieved **89.47% precision, 94.44% recall, and 91.89% micro F1** on the
reported annotations. The single reported loitering event was detected.

Before treating the milestone as fully validated:

- Confirm the full unit-test suite and cleanup edge cases.
- Add event timestamps/track references for A's FN and B's two FP.
- Document the intended out-of-frame ROI policy for C.

Next experiments should address stable ROI entry/exit transitions and add more
positive loitering scenarios, including occlusion, re-entry, and threshold-boundary
cases. Week 5 can then focus on configuration, pipeline organization, and structured
logging while retaining these results as the pre-refactor baseline.
