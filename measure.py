"""
fiber_analysis.measure
----------------------
For each traced skeleton path, sample the fluorescence signal from
every channel along the path, assign a channel label to each pixel,
and aggregate into contiguous segments with physical lengths.

Channel assignment
------------------
* For each pixel on the path we extract the intensity in every channel
  (bilinear sampling from the float32 images).
* We apply a per-channel threshold (default: 3× channel noise σ above
  channel background, estimated from the lower quartile) to produce a
  binary "active / not active" flag per channel.
* Where more than one channel is above threshold we assign the one with
  the highest relative intensity (normalised by its own threshold).
* Where no channel is above threshold the pixel is labelled "unlabelled".

Segment summary per fiber
--------------------------
Consecutive pixels with the same label form a *segment*.  For each segment
we record:
  * channel label
  * length in pixels
  * length in µm   (pixel_length × √2-corrected × um_per_px)
  * start / end coordinates (row, col)

The √2 correction accounts for diagonal steps in the skeleton being
~1.41 px long; we count each step's true Euclidean distance.

Returns
-------
For each fiber, a list of segment dicts and a summary dict.
The public function ``measure_fibers`` returns a flat pandas DataFrame.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure_fibers(
    paths: List[np.ndarray],
    stack: np.ndarray,
    channel_names: List[str],
    um_per_px: float,
    channel_thresholds: Optional[Sequence[float]] = None,
    min_segment_length_um: float = 0.5,
    smooth_profile_sigma: float = 2.0,
) -> pd.DataFrame:
    """Measure each fiber path and return a tidy segment DataFrame.

    Parameters
    ----------
    paths : list of (N, 2) int32 arrays
        Ordered (row, col) pixel paths from ``trace_skeleton``.
    stack : np.ndarray
        (C, H, W) float32 preprocessed image.
    channel_names : list[str]
        Channel labels, e.g. ['CldU', 'IdU'].
    um_per_px : float
        Physical pixel size in µm.
    channel_thresholds : list[float], optional
        Per-channel intensity thresholds in [0, 1].  If None, auto-estimated
        from the 25th-percentile + 3×MAD of each channel.
    min_segment_length_um : float
        Segments shorter than this (µm) are merged with a neighbour or
        discarded.  Prevents tiny label-flicker from fragmenting a fiber.
    smooth_profile_sigma : float
        Gaussian σ (pixels) applied to the sampled intensity profiles before
        thresholding.  Reduces noise-driven label switches.

    Returns
    -------
    df : pd.DataFrame
        One row per segment, columns:
        fiber_id, segment_id, channel, length_px, length_um,
        start_row, start_col, end_row, end_col,
        fiber_total_length_um, n_segments.
    """
    if channel_thresholds is None:
        channel_thresholds = [_auto_threshold(stack[c]) for c in range(stack.shape[0])]

    records = []
    for fid, path in enumerate(paths):
        segments = _measure_one_fiber(
            path, stack, channel_names, um_per_px,
            channel_thresholds, min_segment_length_um, smooth_profile_sigma,
        )
        fiber_length = sum(s["length_um"] for s in segments)
        for sid, seg in enumerate(segments):
            records.append({
                "fiber_id"             : fid,
                "segment_id"           : sid,
                "channel"              : seg["channel"],
                "length_px"            : seg["length_px"],
                "length_um"            : seg["length_um"],
                "start_row"            : seg["start_row"],
                "start_col"            : seg["start_col"],
                "end_row"              : seg["end_row"],
                "end_col"              : seg["end_col"],
                "fiber_total_length_um": fiber_length,
                "n_segments"           : len(segments),
            })

    if not records:
        return pd.DataFrame(columns=[
            "fiber_id","segment_id","channel","length_px","length_um",
            "start_row","start_col","end_row","end_col",
            "fiber_total_length_um","n_segments",
        ])

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _auto_threshold(channel: np.ndarray) -> float:
    """Estimate a signal threshold as Q25 + 3 × MAD."""
    q25 = float(np.percentile(channel, 25))
    mad = float(np.median(np.abs(channel - np.median(channel)))) * 1.4826
    return min(q25 + 3 * mad, 0.5)   # cap at 0.5 to avoid over-thresholding


def _sample_path(
    channel: np.ndarray,
    path: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation along path coordinates from a 2-D channel image."""
    rows = path[:, 0].astype(np.float64)
    cols = path[:, 1].astype(np.float64)
    # map_coordinates expects (axis0, axis1) = (row, col)
    return map_coordinates(channel, [rows, cols], order=1, mode="nearest").astype(np.float32)


def _path_lengths(path: np.ndarray) -> np.ndarray:
    """Return cumulative Euclidean distance along the path (length N)."""
    diff   = np.diff(path.astype(np.float32), axis=0)       # (N-1, 2)
    steps  = np.sqrt((diff ** 2).sum(axis=1))                # Euclidean
    return np.concatenate([[0.0], np.cumsum(steps)])


def _measure_one_fiber(
    path       : np.ndarray,
    stack      : np.ndarray,
    ch_names   : List[str],
    um_per_px  : float,
    thresholds : Sequence[float],
    min_seg_um : float,
    smooth_sig : float,
) -> List[Dict]:
    """Return a list of segment dicts for a single fiber path."""
    from scipy.ndimage import gaussian_filter1d

    n_ch = stack.shape[0]

    # Sample and (optionally) smooth intensity profiles
    profiles = np.zeros((n_ch, len(path)), dtype=np.float32)
    for c in range(n_ch):
        prof = _sample_path(stack[c], path)
        if smooth_sig > 0:
            prof = gaussian_filter1d(prof, sigma=smooth_sig)
        profiles[c] = prof

    # Per-pixel channel assignment
    # Each pixel gets the channel with the highest *relative* signal
    # (signal / threshold), provided it exceeds 1.0.
    rel = np.zeros((n_ch, len(path)), dtype=np.float32)
    for c in range(n_ch):
        t = thresholds[c]
        rel[c] = profiles[c] / max(t, 1e-9)

    best_rel   = rel.max(axis=0)                  # (N,)
    best_chan  = rel.argmax(axis=0)               # (N,) index
    above_thr  = best_rel >= 1.0                  # at least one channel active

    labels = np.where(above_thr, best_chan, -1)   # -1 = unlabelled

    # Compute cumulative length in µm
    cum_len_px  = _path_lengths(path)
    total_len_um = float(cum_len_px[-1] * um_per_px)

    # Collect raw segments
    raw_segs = _run_length_encode(labels, path, cum_len_px, um_per_px, ch_names)

    # Merge very short segments with neighbours
    merged = _merge_short_segments(raw_segs, min_seg_um)

    return merged


def _run_length_encode(
    labels    : np.ndarray,
    path      : np.ndarray,
    cum_len_px: np.ndarray,
    um_per_px : float,
    ch_names  : List[str],
) -> List[Dict]:
    """Convert a per-pixel label array into segment dicts."""
    segments = []
    start    = 0

    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            lbl  = int(labels[start])
            name = ch_names[lbl] if (0 <= lbl < len(ch_names)) else "unlabelled"
            l_px = float(cum_len_px[i - 1] - cum_len_px[start])
            l_um = l_px * um_per_px
            segments.append({
                "channel"  : name,
                "length_px": l_px,
                "length_um": l_um,
                "start_row": int(path[start, 0]),
                "start_col": int(path[start, 1]),
                "end_row"  : int(path[i - 1, 0]),
                "end_col"  : int(path[i - 1, 1]),
                "_start_ix": start,
                "_end_ix"  : i - 1,
            })
            start = i
    return segments


def _merge_short_segments(
    segs      : List[Dict],
    min_um    : float,
) -> List[Dict]:
    """Merge segments shorter than min_um into their longer neighbour."""
    if not segs:
        return segs

    changed = True
    while changed:
        changed = False
        merged  = []
        i = 0
        while i < len(segs):
            s = segs[i]
            if s["length_um"] < min_um and len(segs) > 1:
                # Merge with the longer neighbour
                if i == 0:
                    target = segs[i + 1]
                elif i == len(segs) - 1:
                    target = segs[i - 1]
                else:
                    # pick the longer neighbour
                    target = segs[i + 1] if segs[i + 1]["length_um"] >= segs[i - 1]["length_um"] else segs[i - 1]

                # Absorb: give its length to the neighbour (winner takes all)
                target["length_um"] += s["length_um"]
                target["length_px"] += s["length_px"]
                # Adjust start/end coords
                if i == 0 or (i < len(segs) - 1 and target is segs[i + 1]):
                    target["start_row"] = s["start_row"]
                    target["start_col"] = s["start_col"]
                else:
                    target["end_row"] = s["end_row"]
                    target["end_col"] = s["end_col"]

                segs = [x for j, x in enumerate(segs) if j != i]
                changed = True
                break
            else:
                merged.append(s)
                i += 1
        if not changed:
            segs = merged

    # Remove internal bookkeeping keys
    for s in segs:
        s.pop("_start_ix", None)
        s.pop("_end_ix",   None)

    return segs
