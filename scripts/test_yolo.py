from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model.predict("https://ultralytics.com/images/bus.jpg", verbose=False)

print(results[0].boxes)