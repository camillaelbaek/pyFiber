"""
fiber_analysis.qupath
---------------------
Export fiber and segment annotations as GeoJSON that can be dragged
directly onto a QuPath project entry for instant overlay.

QuPath GeoJSON format
---------------------
QuPath reads GeoJSON feature collections.  Each feature maps to one
QuPath object.  The relevant ``properties`` keys are:

  objectType      : "annotation" | "detection"
  classification  : {name: str, colorRGB: int}  (signed 32-bit ARGB)
  name            : str   (shown in the annotation list)
  isLocked        : bool
  measurements    : [{"name": str, "value": float}, ...]

Geometry types used here:
  LineString  — fiber backbone and individual segments
  Point       — optional: segment midpoint markers

Coordinate convention
---------------------
QuPath pixel coordinates are (x=col, y=row), matching standard image
display axes.  NumPy arrays use (row, col), so we flip before writing.

Color palette
-------------
Fiber backbones    : medium grey
Channel 0 (pulse1): red-ish   (#D62728)
Channel 1 (pulse2): green-ish (#2CA02C)
Additional channels: matplotlib tab10 palette
unlabelled         : light grey

Classification colors in QuPath's signed-int ARGB format:
  fully opaque ⇒ top byte = 0xFF ⇒ value = -(2^24) + RGB_decimal
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Color table
# ---------------------------------------------------------------------------

# Named colors as (R, G, B) tuples
_NAMED_COLORS: Dict[str, tuple] = {
    "CldU"       : (214,  39,  40),   # tab:red
    "IdU"        : ( 44, 160,  44),   # tab:green
    "ch0"        : (214,  39,  40),
    "ch1"        : ( 44, 160,  44),
    "ch2"        : ( 31, 119, 180),   # tab:blue
    "ch3"        : (255, 127,  14),   # tab:orange
    "unlabelled" : (180, 180, 180),
    "fiber"      : (120, 120, 120),
}

_TAB10 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def _rgb_to_qupath_int(r: int, g: int, b: int, alpha: int = 255) -> int:
    """Convert (R, G, B [, A]) to QuPath's signed 32-bit ARGB integer."""
    argb = (alpha << 24) | (r << 16) | (g << 8) | b
    # Convert unsigned to signed 32-bit
    if argb >= 2**31:
        argb -= 2**32
    return argb


def _channel_color(name: str, idx: int) -> int:
    """Return a QuPath color int for a channel by name or index."""
    if name in _NAMED_COLORS:
        return _rgb_to_qupath_int(*_NAMED_COLORS[name])
    rgb = _TAB10[idx % len(_TAB10)]
    return _rgb_to_qupath_int(*rgb)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_geojson(
    paths: List[np.ndarray],
    df: pd.DataFrame,
    channel_names: List[str],
    include_fiber_backbones: bool = True,
    include_segment_lines: bool = True,
    min_segment_length_um: float = 0.0,
) -> dict:
    """Build a GeoJSON FeatureCollection from fiber paths and measurements.

    Parameters
    ----------
    paths : list of (N, 2) int32 arrays
        Ordered (row, col) pixel paths from ``trace_skeleton``.
    df : pd.DataFrame
        Output from ``classify_fibers`` — one row per segment.
    channel_names : list[str]
        Human-readable channel labels.
    include_fiber_backbones : bool
        Add one LineString per fiber showing its full skeleton path.
    include_segment_lines : bool
        Add one LineString per segment, coloured by channel.
    min_segment_length_um : float
        Skip segments shorter than this (µm) from the GeoJSON output.

    Returns
    -------
    dict
        A GeoJSON FeatureCollection dict (serialisable with json.dumps).
    """
    features = []

    # Index segment data by fiber_id for fast lookup
    by_fiber: Dict[int, pd.DataFrame] = {}
    if not df.empty:
        for fid, grp in df.groupby("fiber_id"):
            by_fiber[int(fid)] = grp

    for fid, path in enumerate(paths):
        fiber_data = by_fiber.get(fid, pd.DataFrame())

        # ---- Fiber backbone ----------------------------------------
        if include_fiber_backbones:
            feat = _fiber_backbone_feature(fid, path, fiber_data)
            features.append(feat)

        # ---- Segment lines -----------------------------------------
        if include_segment_lines and not fiber_data.empty:
            seg_feats = _segment_features(
                fid, path, fiber_data, channel_names, min_segment_length_um
            )
            features.extend(seg_feats)

    return {"type": "FeatureCollection", "features": features}


def write_geojson(
    geojson_dict: dict,
    output_path: str | Path,
    indent: Optional[int] = None,
) -> None:
    """Write a GeoJSON dict to disk.

    Parameters
    ----------
    geojson_dict : dict
        Output from ``build_geojson``.
    output_path : str or Path
        Destination file (typically ``*.geojson`` or ``*.json``).
    indent : int, optional
        JSON indentation for human-readable output.  None = compact.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(geojson_dict, fh, indent=indent)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def _path_to_coords(path: np.ndarray) -> list:
    """Convert (N, 2) (row, col) array → [[col, row], …] for GeoJSON."""
    # GeoJSON / QuPath use (x=col, y=row)
    return [[int(c), int(r)] for r, c in path]


def _fiber_backbone_feature(
    fid: int,
    path: np.ndarray,
    fiber_data: pd.DataFrame,
) -> dict:
    """One LineString feature for the full fiber backbone."""
    fiber_color = _rgb_to_qupath_int(*_NAMED_COLORS["fiber"])

    # Collect scalar measurements from the first segment row
    measurements = []
    if not fiber_data.empty:
        row0 = fiber_data.iloc[0]
        _add_m(measurements, "fiber_total_length_um",
               row0.get("fiber_total_length_um"))
        _add_m(measurements, "n_segments",
               row0.get("n_segments"))
        _add_m(measurements, "ratio_2nd_1st",
               row0.get("ratio_2nd_1st"))

    fiber_class = (
        fiber_data.iloc[0]["fiber_class"]
        if (not fiber_data.empty and "fiber_class" in fiber_data.columns)
        else "unknown"
    )

    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": _path_to_coords(path),
        },
        "properties": {
            "objectType"    : "annotation",
            "name"          : f"fiber_{fid:04d}",
            "isLocked"      : False,
            "classification": {
                "name"    : f"Fiber_{fiber_class}",
                "colorRGB": fiber_color,
            },
            "measurements"  : measurements,
        },
    }


def _segment_features(
    fid         : int,
    path        : np.ndarray,
    fiber_data  : pd.DataFrame,
    ch_names    : List[str],
    min_um      : float,
) -> List[dict]:
    """One LineString feature per channel segment within a fiber."""
    features = []
    ch_idx   = {name: i for i, name in enumerate(ch_names)}

    for _, seg in fiber_data.sort_values("segment_id").iterrows():
        if float(seg.get("length_um", 0)) < min_um:
            continue

        ch   = seg["channel"]
        cidx = ch_idx.get(ch, len(ch_names))  # unknown channel → end of tab10
        col  = _channel_color(ch, cidx)

        # Build a sub-path for this segment
        # We use the start/end pixel coords; interpolate if path is available
        seg_path = _slice_path(
            path,
            int(seg["start_row"]), int(seg["start_col"]),
            int(seg["end_row"]),   int(seg["end_col"]),
        )

        measurements = [
            {"name": "length_um",  "value": float(seg["length_um"])},
            {"name": "length_px",  "value": float(seg["length_px"])},
            {"name": "segment_id", "value": float(seg["segment_id"])},
        ]

        features.append({
            "type": "Feature",
            "geometry": {
                "type"       : "LineString",
                "coordinates": _path_to_coords(seg_path),
            },
            "properties": {
                "objectType"    : "annotation",
                "name"          : f"fiber_{fid:04d}_seg_{int(seg['segment_id']):02d}_{ch}",
                "isLocked"      : False,
                "classification": {
                    "name"    : ch,
                    "colorRGB": col,
                },
                "measurements"  : measurements,
            },
        })

    return features


def _slice_path(
    path      : np.ndarray,
    start_row : int, start_col : int,
    end_row   : int, end_col   : int,
) -> np.ndarray:
    """Extract the sub-path between two (row, col) coordinate pairs.

    Falls back to a two-point line if the coordinates are not found in path.
    """
    if len(path) == 0:
        return np.array([[start_row, start_col], [end_row, end_col]], dtype=np.int32)

    # Find closest indices in the path
    def closest_idx(r, c):
        dists = (path[:, 0] - r) ** 2 + (path[:, 1] - c) ** 2
        return int(np.argmin(dists))

    i0 = closest_idx(start_row, start_col)
    i1 = closest_idx(end_row,   end_col)

    if i0 > i1:
        i0, i1 = i1, i0

    sub = path[i0 : i1 + 1]
    if len(sub) < 2:
        sub = np.array([[start_row, start_col], [end_row, end_col]], dtype=np.int32)
    return sub


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _add_m(lst: list, name: str, value) -> None:
    """Append a measurement dict if value is finite."""
    try:
        v = float(value)
        if np.isfinite(v):
            lst.append({"name": name, "value": v})
    except (TypeError, ValueError):
        pass
