import cv2
import torch
import ultralytics

print(f"OpenCV: {cv2.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"Ultralytics: {ultralytics.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")