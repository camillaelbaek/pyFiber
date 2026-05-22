"""
fiber_analysis.io
-----------------
Load (multi-channel) TIFF files produced by widefield or confocal
microscopes.  Tries to extract physical pixel size from OME-TIFF XML;
falls back to a user-supplied value when metadata is absent.

Returned image array is always  (C, H, W)  float32, scaled 0–1 per channel.

Typical usage
-------------
>>> stack, meta = load_tiff("slide01.tif", um_per_px=0.108)
>>> ch_red   = stack[0]   # CldU / Alexa594
>>> ch_green = stack[1]   # IdU  / Alexa488
"""

from __future__ import annotations

import warnings
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tiff(
    path: str | Path,
    channel_axis: Optional[int] = None,
    channel_names: Optional[list[str]] = None,
    um_per_px: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Load a TIFF file and return a (C, H, W) float32 array plus metadata.

    Parameters
    ----------
    path : str or Path
        Path to the TIFF / OME-TIFF file.
    channel_axis : int, optional
        Axis that corresponds to channels in the raw array.  If None the
        function tries to auto-detect from the TIFF axes string.
    channel_names : list[str], optional
        Human-readable labels for each channel, e.g. ['CldU', 'IdU'].
        Defaults to ['ch0', 'ch1', …].
    um_per_px : float, optional
        Physical pixel size in µm/px.  When provided this overrides any
        value found in the metadata.

    Returns
    -------
    stack : np.ndarray
        (C, H, W) float32 array, each channel independently scaled 0–1.
    meta : dict
        Keys: ``um_per_px``, ``channel_names``, ``n_channels``,
              ``height``, ``width``, ``source_path``, ``axes``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with tifffile.TiffFile(str(path)) as tif:
        raw   = tif.asarray()                 # read full array
        axes  = tif.series[0].axes if tif.series else ""
        px_um = _extract_pixel_size(tif)

    # ------------------------------------------------------------------
    # Normalise to (C, H, W)
    # ------------------------------------------------------------------
    raw = raw.squeeze()

    if raw.ndim == 2:                         # single grayscale → 1 channel
        raw = raw[np.newaxis]
    elif raw.ndim == 3:
        raw = _ensure_chw(raw, axes, channel_axis)
    elif raw.ndim == 4:
        # e.g. (Z, C, H, W) or (C, Z, H, W) — take max-projection over Z
        if "Z" in axes.upper():
            z_ax = axes.upper().index("Z")
            raw  = raw.max(axis=z_ax)
        raw = _ensure_chw(raw, axes.replace("Z", ""), channel_axis)

    n_channels = raw.shape[0]

    # ------------------------------------------------------------------
    # Scale each channel to float32 [0, 1]
    # ------------------------------------------------------------------
    stack = np.zeros(raw.shape, dtype=np.float32)
    for c in range(n_channels):
        ch  = raw[c].astype(np.float32)
        lo  = ch.min()
        hi  = ch.max()
        stack[c] = (ch - lo) / (hi - lo + 1e-9)

    # ------------------------------------------------------------------
    # Override pixel size if user supplied one
    # ------------------------------------------------------------------
    if um_per_px is not None:
        px_um = float(um_per_px)
    if px_um is None:
        warnings.warn(
            "Could not extract pixel size from TIFF metadata; "
            "please supply um_per_px=<value>.  Defaulting to 1.0 µm/px.",
            stacklevel=2,
        )
        px_um = 1.0

    # ------------------------------------------------------------------
    # Channel names
    # ------------------------------------------------------------------
    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]
    elif len(channel_names) != n_channels:
        warnings.warn(
            f"channel_names has {len(channel_names)} entries but image has "
            f"{n_channels} channels — truncating / padding.",
            stacklevel=2,
        )
        channel_names = (
            list(channel_names)[:n_channels]
            + [f"ch{i}" for i in range(len(channel_names), n_channels)]
        )

    meta = dict(
        um_per_px     = px_um,
        channel_names = channel_names,
        n_channels    = n_channels,
        height        = stack.shape[1],
        width         = stack.shape[2],
        source_path   = str(path),
        axes          = axes,
    )
    return stack, meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pixel_size(tif: tifffile.TiffFile) -> Optional[float]:
    """Try to read physical pixel size (µm/px) from TIFF tags or OME-XML."""

    # --- OME-TIFF XML ---
    if tif.is_ome:
        try:
            desc = tif.pages[0].description
            m = re.search(
                r'PhysicalSizeX\s*=\s*"([0-9.eE+\-]+)"', desc
            )
            if m:
                return float(m.group(1))
        except Exception:
            pass

    # --- ImageJ / Fiji metadata ---
    try:
        ij = tif.imagej_metadata
        if ij and "unit" in ij:
            unit = ij.get("unit", "")
            spacing = ij.get("spacing", None)
            if spacing and unit.lower() in ("um", "µm", "micron", "micrometer"):
                return float(spacing)
    except Exception:
        pass

    # --- Standard TIFF XResolution tag ---
    try:
        page = tif.pages[0]
        xres = page.tags.get("XResolution")
        unit = page.tags.get("ResolutionUnit")
        if xres:
            num, den = xres.value
            if den == 0:
                return None
            res_per_unit = num / den          # pixels per unit
            if unit and unit.value == 3:      # CENTIMETER
                return 10_000 / res_per_unit  # cm → µm
            elif unit and unit.value == 2:    # INCH
                return 25_400 / res_per_unit  # inch → µm
    except Exception:
        pass

    return None


def _ensure_chw(
    arr: np.ndarray,
    axes: str,
    channel_axis: Optional[int],
) -> np.ndarray:
    """Move the channel axis to position 0, returning (C, H, W)."""
    axes = axes.upper()

    if channel_axis is not None:
        return np.moveaxis(arr, channel_axis, 0)

    # Try to find 'C' in axes string
    if "C" in axes:
        c_ax = axes.index("C")
        return np.moveaxis(arr, c_ax, 0)

    # Heuristic: the *smallest* axis is probably channels
    smallest = int(np.argmin(arr.shape))
    return np.moveaxis(arr, smallest, 0)
