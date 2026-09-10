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
##Usage
python -m scripts.run_video --source data/videos/test.mp4
#Example 
python -m scripts.run_video --source data/videos/test.mp4 --conf 0.4 --output data/outputs/output.mp4
