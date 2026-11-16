"""Experimental view-specific luminance matching for centered cutter crops.

The original crop is never modified. This module produces a separate comparison
image and adjusts CIE L* while retaining the LAB chroma channels.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


REFERENCE_MEDIAN_LSTAR = {
    "top": 50.553,
    "face": 42.197,
}
MAX_CORRECTION_LSTAR = 8.0
SHADOW_HIGHLIGHT_SHOULDER_LSTAR = 12.0


def _lstar_channel(lab: np.ndarray) -> np.ndarray:
    """Decode OpenCV's 8-bit CIE LAB L channel to perceptual L* (0–100)."""
    return lab[:, :, 0].astype(np.float32) * (100.0 / 255.0)


def match_crop_luminance(
    image: Image.Image,
    view: str,
) -> tuple[Image.Image, dict[str, float]]:
    """Return an experimental L*-matched copy and its before/after statistics.

    A capped median shift is applied to L* only. A smooth shoulder protects
    deep shadows and bright highlights from receiving the full adjustment.
    Isometric is intentionally unsupported until a reference crop is selected.
    """
    if view not in REFERENCE_MEDIAN_LSTAR:
        raise ValueError(f"No luminance reference is configured for view: {view}")

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness = _lstar_channel(lab)
    before_median = float(np.median(lightness))
    target = REFERENCE_MEDIAN_LSTAR[view]
    requested_shift = target - before_median
    applied_shift = float(np.clip(
        requested_shift, -MAX_CORRECTION_LSTAR, MAX_CORRECTION_LSTAR
    ))

    # Fade the correction smoothly inside the deepest 12 L* and brightest
    # 12 L* so the adjustment does not crush shadows or flatten highlights.
    shoulder = np.clip(
        np.minimum(
            lightness / SHADOW_HIGHLIGHT_SHOULDER_LSTAR,
            (100.0 - lightness) / SHADOW_HIGHLIGHT_SHOULDER_LSTAR,
        ),
        0.0,
        1.0,
    )
    shoulder = shoulder * shoulder * (3.0 - 2.0 * shoulder)
    adjusted = np.clip(lightness + applied_shift * shoulder, 0.0, 100.0)

    lab_adjusted = lab.copy()
    lab_adjusted[:, :, 0] = np.rint(adjusted * (255.0 / 100.0)).astype(np.uint8)
    matched_rgb = cv2.cvtColor(lab_adjusted, cv2.COLOR_LAB2RGB)

    result = {
        "target_median_lstar": float(target),
        "before_median_lstar": before_median,
        "after_median_lstar": float(np.median(_lstar_channel(lab_adjusted))),
        "requested_shift_lstar": float(requested_shift),
        "applied_shift_lstar": applied_shift,
    }
    return Image.fromarray(matched_rgb), result
