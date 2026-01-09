from ultralytics import YOLO

# Load Pretrained YOLOv8 Model (nano - fastest)
model = YOLO('yolov8n-cls.pt')

# Train
model.train(
    data='dataset_yolo',
    epochs=50,
    imgsz=224,
    batch=32,
    device='mps', # Mac GPU
    project='runs',
    name='smartbin_yolo'  
)

print("Training complete !")