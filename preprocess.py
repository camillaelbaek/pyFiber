"""
fiber_analysis.preprocess
-------------------------
Per-channel image pre-processing for DNA fiber images.

Steps applied in order:
  1. White top-hat transform (subtract large-scale background while
     preserving thin bright fibers).
  2. Optional Gaussian smoothing to reduce detector noise.
  3. Optional contrast stretch (clip low / high percentiles → [0, 1]).

All operations work on (C, H, W) float32 arrays and return the same shape.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(
    stack: np.ndarray,
    tophat_radius: int = 25,
    gaussian_sigma: float = 1.0,
    clip_percentile: Tuple[float, float] = (1.0, 99.5),
    subtract_background: bool = True,
) -> np.ndarray:
    """Pre-process a (C, H, W) float32 image stack.

    Parameters
    ----------
    stack : np.ndarray
        (C, H, W) float32 array from ``load_tiff``.
    tophat_radius : int
        Structuring-element radius (pixels) for the white top-hat background
        subtraction.  Should be larger than the widest fiber but smaller than
        the typical background variation.  ~20–40 px works for most fiber
        images acquired at 40–63×.
    gaussian_sigma : float
        Standard deviation of the Gaussian smoothing kernel (pixels).
        Set to 0 to skip.
    clip_percentile : (float, float)
        Lower and upper percentile values used for intensity re-scaling after
        processing.  (1, 99.5) removes outlier pixels.  Set to (0, 100) to
        disable.
    subtract_background : bool
        If False, skip the top-hat step (useful if background is already
        flat or was corrected upstream).

    Returns
    -------
    np.ndarray
        (C, H, W) float32 array, each channel independently scaled 0–1.
    """
    out = np.empty_like(stack, dtype=np.float32)
    for c in range(stack.shape[0]):
        img = stack[c].copy()
        if subtract_background:
            img = _white_tophat(img, radius=tophat_radius)
        if gaussian_sigma > 0:
            img = gaussian_filter(img, sigma=gaussian_sigma).astype(np.float32)
        img = _rescale(img, clip_percentile)
        out[c] = img
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _white_tophat(img: np.ndarray, radius: int) -> np.ndarray:
    """White top-hat = image − morphological opening (ball structuring element).

    Uses a fast uniform-filter approximation instead of a proper disk SE,
    which is adequate for typical fiber widths and much faster.
    """
    # morphological opening ≈ erosion then dilation; approximated here as
    # a minimum-filter followed by maximum-filter via scipy's rank filters.
    # For simplicity and speed we use the well-known approximation:
    # background = uniform_filter with large kernel (blurs bright fibers less
    # than a true opening but runs in O(N)).
    size = 2 * radius + 1
    background = uniform_filter(img, size=size)
    tophat = img - background
    tophat = np.clip(tophat, 0, None)
    return tophat.astype(np.float32)


def _rescale(img: np.ndarray, clip_pct: Tuple[float, float]) -> np.ndarray:
    lo, hi = np.percentile(img, clip_pct)
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
