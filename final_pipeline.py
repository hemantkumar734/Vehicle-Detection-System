"""
final_pipeline.py
=================
End-to-end vehicle detection pipeline:
  Clean → Rain → Enhanced → YOLO Detection → Comparison Dashboard

Run:
    python final_pipeline.py

Directory layout expected:
    clean/          ← original images
    rain/           ← synthetic-rain versions (same filenames)
    enhanced/       ← auto-created; enhanced rain images saved here
    output/         ← auto-created; annotated detection images saved here
    results/        ← auto-created; CSV + charts saved here
    model/best.pt   ← YOLOv8 trained weights
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

# ── Import enhancement function from existing module ──────────────────────────
try:
    from enhanced_pipeline import enhance_image, Config
except ImportError:
    print(
        "[ERROR] Cannot import from enhanced_pipeline.py — make sure it is "
        "in the same directory as this script."
    )
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# ✏️  CONFIGURATION — adjust paths here if needed
# ─────────────────────────────────────────────────────────────────────────────
CLEAN_DIR = Path("clean")
RAIN_DIR = Path("rain")
ENHANCED_DIR = Path("enhanced")
OUTPUT_DIR = Path("output")
RESULTS_DIR = Path("results")
MODEL_PATH = Path("model/best.pt")

CONF_THRESHOLD = 0.25  # YOLO confidence threshold
MAX_IMAGES = 5  # process first N paired images

# COCO / custom class names to track (lower-case)
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle", "bicycle", "van"]

# Colour palette for bar chart
PALETTE = {"clean": "#4C9BE8", "rain": "#E87C4C", "enhanced": "#4CE8A0"}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: ensure output directories exist
# ─────────────────────────────────────────────────────────────────────────────
def _make_dirs() -> None:
    for d in [ENHANCED_DIR, OUTPUT_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Enhancement
# ─────────────────────────────────────────────────────────────────────────────
def enhance_images(rain_paths: List[Path]) -> List[Path]:
    """
    Apply image enhancement to every rain image and save to ENHANCED_DIR.

    Args:
        rain_paths: List of rain image Paths.

    Returns:
        List of enhanced image Paths (same order as input).
    """
    logger.info("── Step 1: Enhancement ─────────────────────────────────")
    config = Config()  # use default tuned config from enhanced_pipeline
    enhanced_paths: List[Path] = []

    for rain_path in rain_paths:
        img = cv2.imread(str(rain_path))
        if img is None:
            logger.warning("  Skipping unreadable file: %s", rain_path)
            enhanced_paths.append(None)
            continue

        # enhance_image returns (enhanced_ndarray, ProfileResult)
        enhanced_img, profile = enhance_image(img, config)
        logger.info("  %s → %s", rain_path.name, profile.summary().split("\n")[0])

        out_path = ENHANCED_DIR / rain_path.name
        cv2.imwrite(str(out_path), enhanced_img)
        enhanced_paths.append(out_path)

    logger.info("  Saved %d enhanced images to %s/", len(enhanced_paths), ENHANCED_DIR)
    return enhanced_paths


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — YOLO Detection
# ─────────────────────────────────────────────────────────────────────────────
def run_detection(
    model: YOLO,
    image_path: Path,
    label: str,
    img_index: int,
) -> tuple[np.ndarray, list]:
    """
    Run YOLO on a single image, save annotated result, return detections.

    Args:
        model:      Pre-loaded YOLO model (loaded ONCE in main).
        image_path: Path to the image.
        label:      One of 'clean' | 'rain' | 'enhanced'.
        img_index:  1-based index used in the output filename.

    Returns:
        (annotated_bgr_array, list_of_result_dicts)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        logger.warning("  Unreadable: %s", image_path)
        return np.zeros((100, 100, 3), dtype=np.uint8), []

    results = model.predict(img, conf=CONF_THRESHOLD, verbose=False)
    annotated = results[0].plot()  # BGR array with bboxes drawn

    # Save annotated image
    out_name = f"{label}_img{img_index}.jpg"
    cv2.imwrite(str(OUTPUT_DIR / out_name), annotated)

    # Collect raw detection list [{class_name, confidence, bbox}, ...]
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        detections.append(
            {
                "class": model.names[cls_id].lower(),
                "conf": float(box.conf[0]),
                "xyxy": box.xyxy[0].tolist(),
            }
        )

    return annotated, detections


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Vehicle Counting
# ─────────────────────────────────────────────────────────────────────────────
def count_objects(detections: list) -> Dict[str, int]:
    """
    Count vehicles per class from a detection list.
    Classes absent in the detection are filled with 0.

    Args:
        detections: List of dicts from run_detection().

    Returns:
        {class_name: count, ...} for all VEHICLE_CLASSES.
    """
    counts: Dict[str, int] = {cls: 0 for cls in VEHICLE_CLASSES}
    for det in detections:
        cls = det["class"]
        if cls in counts:
            counts[cls] += 1
        else:
            # non-vehicle class — still record if you want all classes
            counts[cls] = counts.get(cls, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — DataFrame
# ─────────────────────────────────────────────────────────────────────────────
def create_dataframe(all_counts: List[Dict]) -> pd.DataFrame:
    """
    Build comparison DataFrame from collected per-image counts.

    Args:
        all_counts: List of dicts, one per image:
                    {"image": str,
                     "clean": {cls: n}, "rain": {cls: n}, "enhanced": {cls: n}}

    Returns:
        DataFrame with columns like car_clean, car_rain, car_enh, …
        plus image, total_clean, total_rain, total_enh.
    """
    rows = []
    for entry in all_counts:
        row = {"image": entry["image"]}
        for cond, suffix in [("clean", "clean"), ("rain", "rain"), ("enhanced", "enh")]:
            counts = entry[cond]
            for cls in VEHICLE_CLASSES:
                row[f"{cls}_{suffix}"] = counts.get(cls, 0)
            row[f"total_{suffix}"] = sum(counts.values())
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("  Saved comparison CSV → %s", csv_path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Graphs
# ─────────────────────────────────────────────────────────────────────────────
def generate_graphs(df: pd.DataFrame) -> tuple[plt.Figure, plt.Figure]:
    """
    Create grouped bar chart and line chart. Save to results/.

    Returns:
        (bar_fig, line_fig)
    """
    images = df["image"].tolist()
    x = np.arange(len(images))
    width = 0.25

    # ── Grouped Bar Chart ─────────────────────────────────────────────
    bar_fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(x - width, df["total_clean"], width, label="Clean", color=PALETTE["clean"])
    ax1.bar(x, df["total_rain"], width, label="Rain", color=PALETTE["rain"])
    ax1.bar(
        x + width, df["total_enh"], width, label="Enhanced", color=PALETTE["enhanced"]
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(images, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Vehicle Count")
    ax1.set_title("Vehicle Count — Clean vs Rain vs Enhanced")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    bar_fig.tight_layout()
    bar_fig.savefig(RESULTS_DIR / "bar_chart.png", dpi=150)

    # ── Line Chart ────────────────────────────────────────────────────
    line_fig, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(
        images, df["total_clean"], marker="o", label="Clean", color=PALETTE["clean"]
    )
    ax2.plot(images, df["total_rain"], marker="s", label="Rain", color=PALETTE["rain"])
    ax2.plot(
        images, df["total_enh"], marker="^", label="Enhanced", color=PALETTE["enhanced"]
    )
    ax2.set_xticks(range(len(images)))
    ax2.set_xticklabels(images, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Vehicle Count")
    ax2.set_title("Detection Trend Across Images")
    ax2.legend()
    ax2.grid(linestyle="--", alpha=0.4)
    line_fig.tight_layout()
    line_fig.savefig(RESULTS_DIR / "line_chart.png", dpi=150)

    logger.info("  Saved bar_chart.png and line_chart.png → %s/", RESULTS_DIR)
    return bar_fig, line_fig


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Single-Page Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def _thumb(img_bgr: np.ndarray, h: int = 180) -> np.ndarray:
    """Resize image to a fixed height, return RGB."""
    ratio = h / img_bgr.shape[0]
    w = int(img_bgr.shape[1] * ratio)
    resized = cv2.resize(img_bgr, (w, h))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def display_dashboard(
    image_rows: List[
        Dict
    ],  # list of {clean, rain, enhanced, det_clean, det_rain, det_enh}
    df: pd.DataFrame,
) -> None:
    """
    Build and show ONE single-page matplotlib dashboard:

    ┌─────────────────────────────────────────────────────────────┐
    │  SECTION 1: Image Pipeline  (clean → rain → enhanced)       │
    │             Detection outputs beneath each                  │
    ├─────────────────────────────────────────────────────────────┤
    │  SECTION 2: Comparison Table (DataFrame)                    │
    ├─────────────────────────────────────────────────────────────┤
    │  SECTION 3: Bar Chart        │  Line Chart                  │
    └─────────────────────────────────────────────────────────────┘
    """
    n = len(image_rows)
    cols_per_cond = n  # one column per image, per condition
    total_img_cols = cols_per_cond * 3  # clean + rain + enhanced

    fig = plt.figure(figsize=(max(18, total_img_cols * 2.2), 28), facecolor="#1a1a2e")

    # ── GridSpec layout ───────────────────────────────────────────────
    outer = gridspec.GridSpec(
        4,
        1,
        figure=fig,
        height_ratios=[3.5, 3.5, 1.5, 3.0],
        hspace=0.35,
    )

    # ── Shared style ──────────────────────────────────────────────────
    title_kw = dict(color="white", fontsize=11, fontweight="bold")
    section_kw = dict(color="#A0C4FF", fontsize=13, fontweight="bold")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — Image flow (2 rows: input → detected)
    # ══════════════════════════════════════════════════════════════════
    sec1 = gridspec.GridSpecFromSubplotSpec(
        2, total_img_cols, subplot_spec=outer[0], hspace=0.08, wspace=0.04
    )

    section_labels = ["CLEAN"] * n + ["RAIN"] * n + ["ENHANCED"] * n
    row0_imgs = (
        [r["clean"] for r in image_rows]
        + [r["rain"] for r in image_rows]
        + [r["enhanced"] for r in image_rows]
    )
    row1_imgs = (
        [r["det_clean"] for r in image_rows]
        + [r["det_rain"] for r in image_rows]
        + [r["det_enhanced"] for r in image_rows]
    )
    border_colours = (
        [PALETTE["clean"]] * n + [PALETTE["rain"]] * n + [PALETTE["enhanced"]] * n
    )

    for col_idx in range(total_img_cols):
        for row_idx, imgs in enumerate([row0_imgs, row1_imgs]):
            ax = fig.add_subplot(sec1[row_idx, col_idx])
            ax.imshow(_thumb(imgs[col_idx]))
            ax.axis("off")
            for spine in ax.spines.values():
                spine.set_edgecolor(border_colours[col_idx])
                spine.set_linewidth(2)
                spine.set_visible(True)
            if row_idx == 0:
                ax.set_title(
                    f"{section_labels[col_idx]}\nImg {(col_idx % n) + 1}",
                    **title_kw,
                    pad=3,
                )

    # Row labels on the left
    fig.text(0.01, 0.86, "INPUT", **section_kw, va="center", rotation=90)
    fig.text(0.01, 0.78, "DETECTED", **section_kw, va="center", rotation=90)

    # Section header
    fig.text(
        0.5,
        0.97,
        "🚗 VEHICLE DETECTION PIPELINE  —  Clean | Rain | Enhanced",
        color="white",
        fontsize=16,
        fontweight="bold",
        ha="center",
        va="top",
    )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2 — Comparison Table
    # ══════════════════════════════════════════════════════════════════
    ax_table = fig.add_subplot(outer[1])
    ax_table.axis("off")
    ax_table.set_title("📊  Comparison Table", **section_kw, pad=8)

    # Select columns to display (total counts for readability)
    display_cols = (
        ["image", "total_clean", "total_rain", "total_enh"]
        + [f"car_{s}" for s in ["clean", "rain", "enh"]]
        + [f"bus_{s}" for s in ["clean", "rain", "enh"]]
        + [f"truck_{s}" for s in ["clean", "rain", "enh"]]
    )
    display_cols = [c for c in display_cols if c in df.columns]
    sub_df = df[display_cols].copy()

    col_labels = [c.replace("_", "\n") for c in sub_df.columns]
    tbl = ax_table.table(
        cellText=sub_df.values,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.8)

    # Style header row
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#333355")
        if r == 0:
            cell.set_facecolor("#2d2d5e")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#1e1e3a")
            cell.set_text_props(color="white")
        else:
            cell.set_facecolor("#252540")
            cell.set_text_props(color="#ccccff")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3 (spacer label row)
    # ══════════════════════════════════════════════════════════════════
    ax_lbl = fig.add_subplot(outer[2])
    ax_lbl.axis("off")
    ax_lbl.text(
        0.5,
        0.5,
        "📈  Performance Comparison Charts",
        **section_kw,
        ha="center",
        va="center",
        transform=ax_lbl.transAxes,
    )

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4 — Charts (bar + line side by side)
    # ══════════════════════════════════════════════════════════════════
    charts = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3], wspace=0.25)

    images_list = df["image"].tolist()
    x = np.arange(len(images_list))
    w = 0.25

    # Bar chart
    ax_bar = fig.add_subplot(charts[0])
    ax_bar.bar(x - w, df["total_clean"], w, label="Clean", color=PALETTE["clean"])
    ax_bar.bar(x, df["total_rain"], w, label="Rain", color=PALETTE["rain"])
    ax_bar.bar(x + w, df["total_enh"], w, label="Enhanced", color=PALETTE["enhanced"])
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        images_list, rotation=20, ha="right", fontsize=7, color="white"
    )
    ax_bar.set_ylabel("Vehicle Count", color="white")
    ax_bar.set_title("Grouped Bar Chart", color="white", fontweight="bold")
    ax_bar.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax_bar.set_facecolor("#12122a")
    ax_bar.tick_params(colors="white")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3, color="white")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#444466")

    # Line chart
    ax_line = fig.add_subplot(charts[1])
    ax_line.plot(
        images_list,
        df["total_clean"],
        marker="o",
        label="Clean",
        color=PALETTE["clean"],
        linewidth=2,
    )
    ax_line.plot(
        images_list,
        df["total_rain"],
        marker="s",
        label="Rain",
        color=PALETTE["rain"],
        linewidth=2,
    )
    ax_line.plot(
        images_list,
        df["total_enh"],
        marker="^",
        label="Enhanced",
        color=PALETTE["enhanced"],
        linewidth=2,
    )
    ax_line.set_xticks(range(len(images_list)))
    ax_line.set_xticklabels(
        images_list, rotation=20, ha="right", fontsize=7, color="white"
    )
    ax_line.set_ylabel("Vehicle Count", color="white")
    ax_line.set_title(
        "Line Chart — Trend Across Images", color="white", fontweight="bold"
    )
    ax_line.legend(facecolor="#2a2a4a", labelcolor="white", fontsize=8)
    ax_line.set_facecolor("#12122a")
    ax_line.tick_params(colors="white")
    ax_line.grid(linestyle="--", alpha=0.3, color="white")
    for spine in ax_line.spines.values():
        spine.set_edgecolor("#444466")

    # ── Save & show ───────────────────────────────────────────────────
    dash_path = RESULTS_DIR / "dashboard.png"
    fig.savefig(dash_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info("  Dashboard saved → %s", dash_path)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    _make_dirs()

    # ── Discover image pairs ──────────────────────────────────────────
    logger.info("── Discovering image pairs ─────────────────────────────")
    clean_imgs = sorted(
        [
            p
            for p in CLEAN_DIR.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )[:MAX_IMAGES]

    if not clean_imgs:
        logger.error("No images found in %s/ — aborting.", CLEAN_DIR)
        sys.exit(1)

    # Match rain images by name; skip pair if rain counterpart is missing
    pairs: List[tuple[Path, Path]] = []
    for cp in clean_imgs:
        rp = RAIN_DIR / cp.name
        if rp.exists():
            pairs.append((cp, rp))
        else:
            logger.warning("  No rain counterpart for %s — skipped.", cp.name)

    if not pairs:
        logger.error("No clean/rain pairs found — aborting.")
        sys.exit(1)

    logger.info("  Found %d image pair(s).", len(pairs))

    # ── Load YOLO model ONCE ──────────────────────────────────────────
    logger.info("── Loading YOLO model ──────────────────────────────────")
    if not MODEL_PATH.exists():
        logger.error("Model not found at %s", MODEL_PATH)
        sys.exit(1)
    model = YOLO(str(MODEL_PATH))
    logger.info("  Model loaded: %s", MODEL_PATH)

    # ── Step 1: Enhancement ───────────────────────────────────────────
    rain_paths = [rp for _, rp in pairs]
    enhanced_paths = enhance_images(rain_paths)

    # ── Step 2 & 3: Detection + Counting ─────────────────────────────
    logger.info("── Step 2: Detection + Step 3: Counting ────────────────")
    all_counts: List[Dict] = []
    image_rows: List[Dict] = []

    for idx, ((clean_path, rain_path), enh_path) in enumerate(
        zip(pairs, enhanced_paths), start=1
    ):
        logger.info("  Processing image %d/%d: %s", idx, len(pairs), clean_path.name)

        # Detection on all three conditions
        det_clean, dets_clean = run_detection(model, clean_path, "clean", idx)
        det_rain, dets_rain = run_detection(model, rain_path, "rain", idx)
        det_enhanced, dets_enhanced = run_detection(model, enh_path, "enhanced", idx)

        # Count vehicles
        counts_clean = count_objects(dets_clean)
        counts_rain = count_objects(dets_rain)
        counts_enhanced = count_objects(dets_enhanced)

        all_counts.append(
            {
                "image": clean_path.stem,
                "clean": counts_clean,
                "rain": counts_rain,
                "enhanced": counts_enhanced,
            }
        )

        # Store images for dashboard (read clean + rain for thumbnails)
        image_rows.append(
            {
                "clean": cv2.imread(str(clean_path)),
                "rain": cv2.imread(str(rain_path)),
                "enhanced": cv2.imread(str(enh_path)),
                "det_clean": det_clean,
                "det_rain": det_rain,
                "det_enhanced": det_enhanced,
            }
        )

    # ── Step 4: DataFrame & CSV ───────────────────────────────────────
    logger.info("── Step 4: Building comparison table ───────────────────")
    df = create_dataframe(all_counts)
    logger.info("\n%s\n", df.to_string(index=False))

    # ── Step 5: Charts ────────────────────────────────────────────────
    logger.info("── Step 5: Generating charts ───────────────────────────")
    generate_graphs(df)

    # ── Step 6: Single-page dashboard ────────────────────────────────
    logger.info("── Step 6: Rendering dashboard ─────────────────────────")
    display_dashboard(image_rows, df)

    logger.info("✅  Pipeline complete. Results saved in %s/", RESULTS_DIR)


if __name__ == "__main__":
    main()
