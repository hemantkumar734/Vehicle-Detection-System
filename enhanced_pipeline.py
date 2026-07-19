"""
Adaptive Image Enhancement Pipeline — Jupyter Notebook Version
===============================================================
Tuned for rainy night / wet windshield / low-light road images.
Just change INPUT_IMAGE_PATH and run all cells.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from IPython.display import display, Image as IPImage

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ===========================================================================
# ✏️  SET YOUR IMAGE PATH HERE
# ===========================================================================

INPUT_IMAGE_PATH = "/home/hr/Documents/Image_enhacement/test2.jpeg"
OUTPUT_IMAGE_PATH = ""  # leave empty → saves as test1_enhanced.jpeg

# ===========================================================================


# ---------------------------------------------------------------------------
# Enums & Config
# ---------------------------------------------------------------------------


class Condition(str, Enum):
    BLUR = "blur"
    LOW_LIGHT = "low_light"
    FOG = "fog"
    NOISE_OR_RAIN = "noise_or_rain"


@dataclass
class Config:
    # ── Detection thresholds ──────────────────────────────────────────
    blur_thresh: float = 300.0  # raised: rainy images are almost always blurry
    dark_thresh: float = 80.0  # raised: night/rain scenes are dim
    fog_thresh: float = 110.0  # lowered: rain haze triggers earlier
    noise_thresh: float = 3.0  # lowered: catch light rain residual too

    # ── Fog / dehazing ────────────────────────────────────────────────
    dark_channel_patch: int = 15
    fog_omega: float = 0.90
    fog_t0: float = 0.15

    # ── Low-light ─────────────────────────────────────────────────────
    clahe_clip: float = 4.0  # stronger CLAHE (was 2.0)
    clahe_grid: int = 8

    # ── Rain / noise removal ──────────────────────────────────────────
    # fastNlMeans is much stronger than bilateral for rain streaks/drops
    nlm_h: float = 10.0  # luminance denoising strength
    nlm_h_color: float = 10.0  # colour channel denoising strength

    # ── Sharpening ────────────────────────────────────────────────────
    sharpen_strength: float = 1.8  # unsharp mask blend weight (was 1.5)
    sharpen_sigma: float = 2.0  # tighter sigma for crisper edges


@dataclass
class ProfileResult:
    conditions: list[Condition] = field(default_factory=list)
    laplacian_variance: float = 0.0
    mean_brightness: float = 0.0
    dark_channel_mean: float = 0.0
    noise_level: float = 0.0

    @property
    def is_degraded(self) -> bool:
        return bool(self.conditions)

    def summary(self) -> str:
        if not self.conditions:
            return "No degradation detected — image looks clean."
        conds = ", ".join(c.value for c in self.conditions)
        return (
            f"Detected : [{conds}]\n"
            f"  Laplacian variance : {self.laplacian_variance:.2f}\n"
            f"  Mean brightness    : {self.mean_brightness:.2f}\n"
            f"  Dark channel mean  : {self.dark_channel_mean:.2f}\n"
            f"  Noise level        : {self.noise_level:.2f}"
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    logger.info("Loaded  : %s  (%dx%d)", path, img.shape[1], img.shape[0])
    return img


def save_image(image: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Could not save: {path}")
    logger.info("Saved   : %s", path)


def default_output_path(input_path: str | Path) -> Path:
    p = Path(input_path)
    return p.with_stem(p.stem + "_enhanced")


def show_images_notebook(original: np.ndarray, enhanced: np.ndarray) -> None:
    """Show original and enhanced side-by-side inside Jupyter."""
    # Resize both to same height for clean display
    h = min(original.shape[0], enhanced.shape[0], 600)

    def resize_h(img, target_h):
        ratio = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * ratio), target_h))

    orig_r = resize_h(original, h)
    enh_r = resize_h(enhanced, h)

    # Add labels
    label_orig = np.zeros((40, orig_r.shape[1], 3), dtype=np.uint8)
    label_enh = np.zeros((40, enh_r.shape[1], 3), dtype=np.uint8)
    cv2.putText(
        label_orig,
        "ORIGINAL",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        label_enh,
        "ENHANCED",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (100, 255, 100),
        2,
    )

    left = np.vstack([label_orig, orig_r])
    right = np.vstack([label_enh, enh_r])
    combined = np.hstack([left, right])

    _, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
    display(IPImage(data=buf.tobytes()))


# ---------------------------------------------------------------------------
# Degradation Analyzer
# ---------------------------------------------------------------------------


class DegradationAnalyzer:
    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()

    def profile(self, image: np.ndarray) -> ProfileResult:
        self._validate(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = ProfileResult()

        result.laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if result.laplacian_variance < self.cfg.blur_thresh:
            result.conditions.append(Condition.BLUR)

        result.mean_brightness = float(np.mean(gray))
        if result.mean_brightness < self.cfg.dark_thresh:
            result.conditions.append(Condition.LOW_LIGHT)

        dark_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.dark_channel_patch, self.cfg.dark_channel_patch),
        )
        dark_eroded = cv2.erode(dark_channel, kernel)
        result.dark_channel_mean = float(np.mean(dark_eroded))
        if result.dark_channel_mean > self.cfg.fog_thresh:
            result.conditions.append(Condition.FOG)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        result.noise_level = float(np.mean(cv2.absdiff(gray, blurred)))
        if result.noise_level > self.cfg.noise_thresh:
            result.conditions.append(Condition.NOISE_OR_RAIN)

        return result

    @staticmethod
    def _validate(image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("image must be a non-empty ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a 3-channel BGR image (H×W×3).")


# ---------------------------------------------------------------------------
# Adaptive Enhancer
# ---------------------------------------------------------------------------


class AdaptiveEnhancer:
    """
    Enhancement order: LOW_LIGHT → NOISE_OR_RAIN → FOG → BLUR
    """

    _PIPELINE = [
        Condition.LOW_LIGHT,
        Condition.NOISE_OR_RAIN,
        Condition.FOG,
        Condition.BLUR,
    ]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.cfg = config or Config()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_grid, self.cfg.clahe_grid),
        )

    def process(self, image: np.ndarray, profile: ProfileResult) -> np.ndarray:
        out = image.copy()
        for condition in self._PIPELINE:
            if condition in profile.conditions:
                out = self._dispatch(condition)(out)
                logger.info("  ✔ Applied: %s", condition.value)
        return out

    # ── Individual enhancers ──────────────────────────────────────────

    def _enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Two-pass brightening:
          1. Gamma correction to lift shadows globally.
          2. CLAHE on LAB L-channel for local contrast recovery.
        """
        # Gamma lift (gamma < 1 brightens the image)
        gamma = 0.6
        table = np.array(
            [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8
        )
        gamma_corrected = cv2.LUT(image, table)

        # CLAHE on L channel
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Non-local Means denoising — far more effective than bilateral
        for rain streaks and water droplets on glass.
        """
        return cv2.fastNlMeansDenoisingColored(
            image,
            None,
            h=self.cfg.nlm_h,
            hColor=self.cfg.nlm_h_color,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _remove_fog(self, image: np.ndarray) -> np.ndarray:
        """Dark Channel Prior dehazing (He et al., 2009)."""
        img_f = image.astype(np.float64) / 255.0
        dark_ch = np.min(img_f, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.cfg.dark_channel_patch, self.cfg.dark_channel_patch),
        )
        dark_ch = cv2.erode(dark_ch, kernel)
        n_top = max(1, dark_ch.size // 1000)
        flat_idx = np.argsort(dark_ch.ravel())[-n_top:]
        coords = np.unravel_index(flat_idx, dark_ch.shape)
        atm_light = max(float(np.max(img_f[coords])), 1e-6)
        t = np.clip(
            1.0 - self.cfg.fog_omega * dark_ch / atm_light,
            self.cfg.fog_t0,
            1.0,
        )
        t3d = t[:, :, np.newaxis]
        recovered = (img_f - atm_light) / t3d + atm_light
        return np.clip(recovered * 255.0, 0, 255).astype(np.uint8)

    def _sharpen_blur(self, image: np.ndarray) -> np.ndarray:
        """Stronger unsharp masking for rain-blurred edges."""
        blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=self.cfg.sharpen_sigma)
        return cv2.addWeighted(
            image,
            self.cfg.sharpen_strength,
            blurred,
            -(self.cfg.sharpen_strength - 1.0),
            0,
        )

    def _dispatch(self, condition: Condition):
        return {
            Condition.LOW_LIGHT: self._enhance_low_light,
            Condition.NOISE_OR_RAIN: self._remove_noise,
            Condition.FOG: self._remove_fog,
            Condition.BLUR: self._sharpen_blur,
        }[condition]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enhance_image(
    image: np.ndarray,
    config: Optional[Config] = None,
) -> tuple[np.ndarray, ProfileResult]:
    cfg = config or Config()
    profile = DegradationAnalyzer(cfg).profile(image)
    result = (
        AdaptiveEnhancer(cfg).process(image, profile) if profile.is_degraded else image
    )
    return result, profile


# ---------------------------------------------------------------------------
# ▶  RUN — just execute this cell
# ---------------------------------------------------------------------------

image = load_image(INPUT_IMAGE_PATH)
config = Config()
enhanced, profile = enhance_image(image, config)

print(profile.summary())

out_path = (
    OUTPUT_IMAGE_PATH if OUTPUT_IMAGE_PATH else default_output_path(INPUT_IMAGE_PATH)
)
save_image(enhanced, out_path)
print(f"\n✅ Saved to: {out_path}")

show_images_notebook(image, enhanced)
