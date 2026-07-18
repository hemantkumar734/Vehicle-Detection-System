# Vehicle Detection on Indian Roads Under Rainy Weather Using YOLOv8x and Image Enhancement

A robust, weather-aware vehicle detection system built for the chaotic reality of Indian traffic — heterogeneous vehicle types, unstructured roads, and heavy monsoon rain that most detection models were never trained to handle.

📄 **Conference Paper:** *Vehicle Detection on Indian Roads Under Rainy Weather Using YOLOv8x and Image Enhancement Technique*

---

## 🚦 Overview

Standard object detection models are usually trained and benchmarked on clean, structured datasets from Europe, the US, or China — well-marked lanes, predictable traffic, and clear weather. None of that describes an Indian monsoon street: auto-rickshaws weaving between buses, waterlogged roads acting like mirrors, rain streaks blurring object boundaries, and vehicle spray fogging up entire lanes.

This project builds a detection pipeline specifically designed for that environment. It combines a custom-annotated, real-world rainy-weather dataset with a **self-built image enhancement pipeline** and a **SAHI-based multi-inference strategy** on top of YOLOv8x — pushing detection accuracy up significantly compared to a standard baseline model, especially for small and partially occluded vehicles.

---

## 🎯 Problem Statement

Rainy weather degrades vehicle detection accuracy through:
- **Rain streaks** — thin, high-frequency visual noise that obscures object edges
- **Water spray** from passing vehicles, creating localized fog
- **Wet-road reflections** that produce false positives (ghost objects) and false negatives (real vehicles blending into glare)
- **Low contrast and reduced visibility** from overcast skies and ambient moisture
- **Highly heterogeneous traffic** — cars, auto-rickshaws, two-wheelers, buses, and trucks sharing the same unstructured lanes with no consistent discipline

Existing global datasets (Cityscapes, BDD100K, DAWN, ACDC) don't capture this combination of adverse weather *and* Indian road chaos — which is the gap this project addresses with a purpose-built dataset and pipeline.

---

## 🧩 Key Contributions

### 1. Custom Real-World Rainy Dataset
- ~3,000 images collected from mobile cameras, CCTV, and dashcams across Indian urban roads, highways, and congested intersections during actual monsoon conditions
- Filtered down to **1,800 high-quality images** after removing blur, occlusion, and redundant samples
- Manually annotated in **Label Studio** (no AI-assisted labeling) across **6 vehicle classes**: Car, Bus, Truck, Auto-Rickshaw, Two-Wheeler, Mini Truck
- ~10,000 total labeled vehicle instances
- Verified through a custom Python validation script checking bounding box integrity, class IDs, and label consistency, with an iterative annotation feedback loop
- Split 70:20:10 (train:val:test)

### 2. Self Image Enhancement Pipeline
A 3-stage sequential enhancement pipeline applied before detection, designed to preserve object structure (not just generic visual quality):

```
Rain-degraded image → Bilateral Denoising → Gamma Correction → CLAHE → Enhanced image → YOLOv8x
```

- **Bilateral Denoising** — removes rain-induced noise while preserving object edges (unlike standard smoothing, which blurs vehicle contours)
- **Gamma Correction** — non-linear intensity adjustment to recover detail in dark, low-visibility regions common in overcast rain
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — localized contrast enhancement to separate vehicles from hazy/water-sprayed backgrounds

### 3. SAHI-Based 4-Quadrant Inference Strategy
To recover small and distant vehicles that get lost at standard 640×640 inference resolution:
- **Global Baseline Scan** — one full-image pass to catch clearly visible vehicles
- **4-Quadrant Slicing** — image split into top-left, top-right, bottom-left, bottom-right; each processed independently at higher effective resolution (1+4 inference architecture)
- **IoM (Intersection over Minimum) Duplicate Suppression** — resolves duplicate detections between global and quadrant passes more reliably than standard IoU, which fails when a loose global box overlaps a tighter local box
- **Dynamic Per-Class Confidence Thresholds** — lower thresholds for small/faint classes like two-wheelers (to boost recall), higher thresholds for large vehicles like buses/heavy vehicles (to reduce false positives)

---

## 📊 Results

**Model:** YOLOv8x, transfer-learned from COCO pre-trained weights
**Training:** 120 epochs, AdamW optimizer, batch size 4, input size 640×640

| Metric | Value |
|---|---|
| Precision | 0.89 |
| Recall | 0.84 |
| mAP@0.5 | 0.918 |
| mAP@0.5–0.95 | 0.73 |
| Peak F1 Score | 0.86 (at confidence threshold 0.426) |

**Per-class Average Precision (highlights):**
| Class | AP |
|---|---|
| Mini Truck | 0.970 |
| Auto-Rickshaw | 0.936 |
| Car | 0.932 |
| Bus | 0.928 |
| Heavy Vehicle | 0.881 |
| Two-Wheeler | 0.861 |

**Enhancement impact:** Across all evaluated rainy test images, the enhancement pipeline improved recovered detections in *every single case* — no performance regressions observed. Example: one test image went from 37 → 49 detected vehicles after enhancement (+12), another from 6 → 10 (+4) even in a low-yield scene.

**Training/validation loss** converged smoothly across 120 epochs with no significant train-val divergence, indicating the model generalized well without overfitting.

*(See `/output` folder for training curves, confusion matrix, and enhancement comparison visuals.)*

---

## 🛠️ Tech Stack

- **Detection Model:** YOLOv8x (Ultralytics)
- **Enhancement:** OpenCV (Bilateral Filter, Gamma Correction, CLAHE)
- **Inference Strategy:** SAHI (Slicing Aided Hyper Inference), custom IoM suppression logic
- **Annotation:** Label Studio
- **Core Libraries:** Python, OpenCV, NumPy, Pandas, PyTorch
- **Validation:** Custom Python dataset-integrity scripts
- **Performance:** GPU acceleration (CUDA)

---

## 📁 Repository Structure

```
Vehicle-Detection-System/
├── vehicle_detection.py    # Main detection script
├── requirements.txt        # Dependencies
├── test_images/            # Sample test images
└── output/                 # training_results.png, confusion_matrix.png, enhancement_comparison.png
```

---

## 🚀 How to Run

1. Clone the repository
   ```
   git clone https://github.com/hemantkumar734/Vehicle-Detection-System.git
   ```
2. Install dependencies
   ```
   pip install -r requirements.txt
   ```
3. Run detection
   ```
   python vehicle_detection.py
   ```

---

## 🔭 Why This Matters

India recorded ~4.6 lakh road accidents in 2022 (Ministry of Road Transport and Highways), with adverse weather a significant contributing factor. Reliable vehicle detection under rain isn't just a computer vision benchmark — it's a real safety gap for intelligent transportation systems, ADAS, and traffic monitoring deployed in Indian conditions. This project is a step toward detection systems that actually hold up in the environment they'll be deployed in, rather than just the one they were trained on.

---

## 👤 Author

**Hemant Kumar Singh**
Final-year B.Tech, Information Technology & Data Science, Ajeenkya DY Patil University
📧 hemantkumar.s734@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/hemant-singh-3aa705318)
💻 [GitHub](https://github.com/hemantkumar734)
