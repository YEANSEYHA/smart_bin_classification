import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# Setup device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model
checkpoint = torch.load('smartbin_model.pth')
class_names = checkpoint['class_names']

model = models.efficientnet_b0(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(1280, 256),
    nn.ReLU(),
    nn.Linear(256, len(class_names))
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Auto-detect camera
def find_camera():
    print("Searching for cameras...")
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                print(f"  Camera {i}: {w}x{h} ✓")
                available.append((i, w, h))
            cap.release()
    
    if not available:
        print("No cameras found!")
        return None
    
    # Pick highest resolution (likely iPhone)
    best = max(available, key=lambda x: x[1] * x[2])
    print(f"\nUsing Camera {best[0]} ({best[1]}x{best[2]})")
    return best[0]

camera_index = find_camera()
if camera_index is None:
    exit()

cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
print("Press 'q' to quit")

last_prediction_time = 0
prediction_interval = 1
stable_label = "Waiting..."
stable_confidence = 0
stable_color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    box_w = int(w * 0.8)
    box_h = int(h * 0.8)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h
    
    current_time = time.time()
    if current_time - last_prediction_time > prediction_interval:
        roi = frame[y1:y2, x1:x2]
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, predicted = probs.max(1)
        
        stable_label = class_names[predicted.item()]
        stable_confidence = conf.item() * 100
        stable_color = (0, 255, 0) if stable_confidence > 70 else (0, 255, 255)
        last_prediction_time = current_time
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), stable_color, 3)
    cv2.putText(frame, "Place item here", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    text = f"{stable_label}: {stable_confidence:.1f}%"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 3, 6)[0]
    text_x = (w - text_size[0]) // 2
    cv2.putText(frame, text, (text_x, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 3, stable_color, 6)
    
    cv2.imshow('Smart Bin Classifier', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
