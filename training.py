"""
YOLOv8x Training Script
========================
Trains a YOLOv8x model on the custom Indian-roads rainy-weather vehicle
detection dataset (6 classes: Car, Bus, Heavy Vehicle, Auto-Rickshaw,
Two-Wheeler, Mini Truck).

Note: This script was originally run in a Kaggle notebook environment
(hence the /kaggle/working/runs save path and data="combined.yaml"
referencing a Kaggle dataset). If running locally or elsewhere, update
the `data` path and `project` save path to match your own environment.
"""

from ultralytics import YOLO

# Load base model (pre-trained on COCO)
model = YOLO("yolov8x.pt")

model.train(
    data="combined.yaml",

    # Training power
    epochs=120,
    imgsz=640,
    batch=4,

    # Performance
    device=0,
    workers=2,
    amp=True,  # mixed precision (faster + better)

    # Smart training
    patience=30,  # early stopping
    optimizer="AdamW",
    lr0=0.001,

    # Generalization boost
    cos_lr=True,  # cosine learning rate decay
    close_mosaic=10,  # disable mosaic augmentation near the end

    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,

    # Save settings
    save=True,
    save_period=20,  # checkpoint every 20 epochs
    project="/kaggle/working/runs",
    name="yolov8x_best",

    # Logging
    verbose=True,
)
