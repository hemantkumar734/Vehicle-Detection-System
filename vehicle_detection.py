# Vehicle Detection using YOLOv8
# Author: Hemant Kumar Singh
# Description: ML-based vehicle detection system using YOLOv8, OpenCV, and PyTorch

from ultralytics import YOLO
import cv2
import torch

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "weights/last.pt"        # Path to trained YOLOv8 model weights
IMAGE_PATH = "test_images/test.jpg"   # Path to input test image
OUTPUT_PATH = "output/output.jpg"     # Path to save detection result
CONF = 0.3                            # Confidence threshold for detections
IOU = 0.5                             # IoU threshold for Non-Maximum Suppression
MAX_DET = 100                         # Maximum number of detections per image

# -------------------------------
# DEVICE CONFIGURATION
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading YOLOv8 model...")
model = YOLO(MODEL_PATH)
model.to(device)

# -------------------------------
# RUN PREDICTION
# -------------------------------
print("Running vehicle detection...")
results = model.predict(
    source=IMAGE_PATH,
    conf=CONF,
    iou=IOU,
    max_det=MAX_DET,
    device=device,
    save=False,
    verbose=True,
)
print("Detection complete!")

# -------------------------------
# PROCESS AND DISPLAY RESULTS
# -------------------------------
result = results[0]

if result.boxes is not None:
    print(f"\nTotal vehicles detected: {len(result.boxes)}")
    print("\nDetection Details:")
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  Class: {model.names[cls_id]:<15} Confidence: {conf:.2f}")
else:
    print("No vehicles detected.")

# Draw bounding boxes on image
img = result.plot()

# Save output image
cv2.imwrite(OUTPUT_PATH, img)
print(f"\nOutput saved to: {OUTPUT_PATH}")

# Display result
cv2.imshow("Vehicle Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()