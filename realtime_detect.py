import cv2
import numpy as np
from ultralytics import YOLO
import time


def find_camera():
    """Use MacBook built-in FaceTime camera as input; output displays on Mac screen."""
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()

    print(f"Cameras found: {available} — using index {available[0]} (MacBook built-in)")
    return available[0] if available else 0

# Load YOLO classification model
model = YOLO('runs/classify/output/yolo_cls/weights/best.pt')
class_names = model.names  # dict {0: 'class1', 1: 'class2', ...}

# Class colors (BGR) — vivid, distinct per class
CLASS_COLORS = {
    'cardboard': (0, 140, 255),    # orange
    'glass':     (255, 210, 0),    # cyan-yellow
    'metal':     (200, 200, 200),  # silver
    'paper':     (50, 255, 255),   # yellow
    'plastic':   (255, 80, 80),    # blue
    'trash':     (80, 80, 80),     # dark gray
}
DEFAULT_COLOR = (180, 180, 180)
UNCERTAIN_COLOR = (0, 215, 255)  # gold when low confidence

def get_color(label, confidence):
    if confidence < 70:
        return UNCERTAIN_COLOR
    return CLASS_COLORS.get(label.lower(), DEFAULT_COLOR)

def draw_corner_brackets(frame, x1, y1, x2, y2, color, thickness=3, length=30):
    """Draw corner brackets instead of a full rectangle."""
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)

def draw_overlay_panel(frame, x, y, w, h, alpha=0.55):
    """Draw a semi-transparent dark panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_confidence_bar(frame, x, y, bar_w, conf, color):
    """Draw a thin confidence progress bar."""
    bar_h = 6
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)
    filled = int(bar_w * conf / 100)
    cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), color, -1)

# Webcam — prefer iPhone Continuity Camera
cap = cv2.VideoCapture(find_camera())
print("Press 'q' to quit")

last_prediction_time = 0
prediction_interval = 1
stable_label = "Waiting..."
stable_confidence = 0
stable_color = DEFAULT_COLOR

fps_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # FPS
    now = time.time()
    fps = 1.0 / (now - fps_time + 1e-6)
    fps_time = now

    # Detection zone (75% of frame)
    box_w = int(w * 0.75)
    box_h = int(h * 0.75)
    x1 = (w - box_w) // 2
    y1 = (h - box_h) // 2
    x2 = x1 + box_w
    y2 = y1 + box_h

    # Predict every interval
    if now - last_prediction_time > prediction_interval:
        roi = frame[y1:y2, x1:x2]
        results = model(roi, verbose=False)
        probs = results[0].probs
        predicted = int(probs.top1)
        conf = probs.top1conf.item()

        stable_confidence = conf * 100
        stable_label = class_names[predicted] if stable_confidence >= 70 else "other"
        stable_color = get_color(stable_label, stable_confidence)
        last_prediction_time = now

    # ── Top bar ──────────────────────────────────────────────
    draw_overlay_panel(frame, 0, 0, w, 48)
    cv2.putText(frame, "Smart Bin Classifier", (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 215, 255), 2)   # gold title
    fps_text = f"FPS  {fps:.0f}"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.putText(frame, fps_text, (w - fps_size[0] - 14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)  # green FPS

    # ── Detection zone ────────────────────────────────────────
    # Dim area outside the zone slightly
    mask = np.zeros_like(frame)
    mask[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    dimmed = (frame * 0.4).astype(np.uint8)
    dimmed[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    frame[:] = dimmed

    draw_corner_brackets(frame, x1, y1, x2, y2, stable_color, thickness=3, length=28)

    # Zone label (top-center of box)
    zone_label = "SCAN AREA"
    zl_size = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    zl_x = x1 + (box_w - zl_size[0]) // 2
    cv2.putText(frame, zone_label, (zl_x, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, stable_color, 1)

    # ── Bottom result panel ───────────────────────────────────
    panel_h = 90
    draw_overlay_panel(frame, 0, h - panel_h, w, panel_h)

    label_upper = stable_label.upper()
    font_scale, thickness = 1.4, 3
    label_size = cv2.getTextSize(label_upper, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]

    # Colored badge behind the label text
    pad_x, pad_y = 18, 8
    badge_x = (w - label_size[0]) // 2 - pad_x
    badge_y = h - panel_h + 10
    badge_x2 = badge_x + label_size[0] + pad_x * 2
    badge_y2 = badge_y + label_size[1] + pad_y * 2
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x2, badge_y2), stable_color, -1)
    cv2.rectangle(frame, (badge_x, badge_y), (badge_x2, badge_y2), (255, 255, 255), 1)

    # Black text on the colored badge
    label_x = (w - label_size[0]) // 2
    cv2.putText(frame, label_upper, (label_x, badge_y2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

    # Confidence % next to badge
    conf_text = f"{stable_confidence:.1f}%"
    ct_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
    cv2.putText(frame, conf_text, (badge_x2 + 10, badge_y2 - pad_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, stable_color, 2)

    # Confidence bar
    bar_w = w - 80
    draw_confidence_bar(frame, 40, h - 14, bar_w, stable_confidence, stable_color)

    cv2.imshow('Smart Bin Classifier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
