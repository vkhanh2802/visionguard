# VisionGuard

VisionGuard is a real-time video analytics project for object detection, multi-object tracking and event understanding.

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

## Pipeline

Video → OpenCV → YOLO26 → Detection → Visualization → Output

## Installation

```bash
conda env create -f environment.yml
conda activate visionguard
```
## Usage 
python -m scripts.run_video --source data/videos/test.mp4
## Example 
python -m scripts.run_video --source data/videos/test.mp4 --conf 0.4 --output data/outputs/output.mp4

## Week 2 — Multi-Object Tracking

Implemented:

- ByteTrack integration
- Persistent object IDs
- Track abstraction
- Centroid and bottom-center extraction
- Track trajectory visualization
- Track history cleanup
- IoU implementation
- Geometry unit tests
# Tracking Confidence Experiment

To evaluate the effect of detection confidence on tracking stability, the same crowded video was tested with different confidence thresholds.

# Observation

With `conf=0.2`, the tracker generated new track IDs much more frequently. When a person was temporarily missed because of occlusion or weak detection, the person often reappeared with a new and significantly larger track ID. The overall track ID count also increased very quickly, indicating a larger number of short-lived or noisy tracks.

With `conf=0.4`, track IDs still increased over time, which is expected when new objects enter the scene, but the increase was noticeably slower. More importantly, some people who were temporarily missed were successfully associated with their previous track ID when they reappeared. This behavior was less stable with `conf=0.2`.

# Comparison

| Confidence | Tracking behavior                                                                                                                  |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `0.2`      | More weak detections, rapid growth in track IDs, more short-lived tracks, and more frequent ID reassignment after temporary misses |
| `0.4`      | Fewer noisy tracks, slower ID growth, and better ID continuity for several objects after short occlusions                          |

# Conclusion

For the tested crowded scene, lowering the confidence threshold did not improve tracking quality. Although a lower threshold provides more detections to ByteTrack, it also introduces additional low-confidence and noisy detections that can interfere with track association and create unnecessary new tracks.

In this experiment, `conf=0.4` provided a better balance between detection coverage and tracking stability. It produced fewer fragmented tracks and showed better identity continuity after temporary missed detections.

Therefore, `0.4` is currently used as the default detection confidence threshold for VisionGuard. This value is treated as a scene-dependent hyperparameter rather than a universal optimal threshold and may be adjusted for different camera conditions or datasets.
