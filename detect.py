"""
fiber_analysis.detect
---------------------
Build a binary fiber mask from a pre-processed (C, H, W) image stack and
return the corresponding skeleton image.

Strategy
--------
1. Combine all (or selected) channels by maximum projection → grayscale.
2. Auto-threshold using Otsu's method (with an optional multiplicative
   factor to fine-tune sensitivity).
3. Remove small objects and fill small holes.
4. Optionally apply a Frangi vesselness filter to suppress blobs / nuclei
   and enhance elongated structures before thresholding.
5. Skeletonise the mask → one-pixel-wide skeleton.

The skeleton is the input for ``skeleton.py``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import binary_fill_holes, label as scipy_label
from skimage.filters import threshold_otsu, frangi
from skimage.morphology import (
    closing as morphological_closing,
    binary_dilation,
    remove_small_objects,
    disk,
    skeletonize,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_fiber_mask(
    stack: np.ndarray,
    channel_indices: Optional[Sequence[int]] = None,
    otsu_factor: float = 0.8,
    min_object_area: int = 200,
    close_radius: int = 2,
    use_frangi: bool = False,
    frangi_scale_range: tuple = (1, 4),
) -> np.ndarray:
    """Create a binary fiber mask from a pre-processed (C, H, W) stack.

    Parameters
    ----------
    stack : np.ndarray
        (C, H, W) float32 array.
    channel_indices : list[int], optional
        Which channels to combine.  Defaults to all channels.
    otsu_factor : float
        Multiply the Otsu threshold by this factor.  Values < 1 make the
        threshold more permissive (keep more signal); > 1 makes it stricter.
        0.8 works well for most fiber images.
    min_object_area : int
        Minimum connected-component area in pixels.  Small spots (nuclei
        bleed-through, dust) below this are discarded.
    close_radius : int
        Radius of disk SE used for binary closing to bridge tiny gaps in
        fibers.
    use_frangi : bool
        If True, apply a Frangi vesselness filter to the combined image
        before thresholding.  Improves discrimination of fibers vs. bright
        blobs but adds ~2–5× compute time.
    frangi_scale_range : (float, float)
        Min and max scale (σ, in pixels) for Frangi.

    Returns
    -------
    mask : np.ndarray
        (H, W) bool array — True where a fiber is present.
    """
    if channel_indices is None:
        channel_indices = list(range(stack.shape[0]))

    # ---- 1. Combine selected channels -----------------------------------
    combined = stack[channel_indices].max(axis=0)   # (H, W) float32

    # ---- 2. Optional Frangi vesselness ---------------------------------
    if use_frangi:
        s_min, s_max = frangi_scale_range
        combined = frangi(
            combined,
            sigmas=np.linspace(s_min, s_max, 4),
            black_ridges=False,
        ).astype(np.float32)
        # rescale to [0, 1]
        hi = combined.max()
        if hi > 0:
            combined /= hi

    # ---- 3. Threshold ---------------------------------------------------
    thr = threshold_otsu(combined) * otsu_factor
    mask = combined > thr

    # ---- 4. Morphological clean-up -------------------------------------
    if close_radius > 0:
        mask = morphological_closing(mask, footprint=disk(close_radius))
    mask = binary_fill_holes(mask)
    # skimage >= 0.26 renamed min_size → max_size (exclusive threshold)
    import inspect as _ins
    _rso_kw = "max_size" if "max_size" in _ins.signature(remove_small_objects).parameters else "min_size"
    mask = remove_small_objects(mask.astype(bool), **{_rso_kw: min_object_area})

    return mask


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Return the one-pixel-wide skeleton of a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        (H, W) bool array.

    Returns
    -------
    skeleton : np.ndarray
        (H, W) bool array — True on skeleton pixels.
    """
    return skeletonize(mask)


def label_fiber_regions(mask: np.ndarray) -> np.ndarray:
    """Label connected components in the fiber mask.

    Returns an int32 label image (0 = background, 1…N = individual fibers).
    """
    labels, _ = scipy_label(mask)
    return labels.astype(np.int32)
