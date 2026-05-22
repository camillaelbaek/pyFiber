"""
fiber_analysis.pipeline
-----------------------
High-level ``run_analysis()`` function that orchestrates the full pipeline:

  load → preprocess → detect mask → skeletonise → trace paths →
  measure segments → classify fibers → build GeoJSON

Typical usage
-------------
>>> from fiber_analysis import run_analysis
>>>
>>> results = run_analysis(
...     tiff_path       = "experiment/slide01_ch00_ch01.tif",
...     channel_names   = ["CldU", "IdU"],
...     um_per_px       = 0.108,            # from your microscope calibration
...     output_dir      = "results/slide01",
... )
>>> results["measurements"].head()
>>> # QuPath overlay: drag results/slide01/slide01_fibers.geojson onto the image

For batch processing see ``run_batch()`` below.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .io        import load_tiff
from .preprocess import preprocess
from .detect    import make_fiber_mask, skeletonize_mask
from .skeleton  import trace_skeleton
from .measure   import measure_fibers
from .classify  import classify_fibers, summarise_by_class
from .qupath    import build_geojson, write_geojson


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_analysis(
    tiff_path                : str | Path,
    channel_names            : Optional[List[str]] = None,
    um_per_px                : Optional[float]     = None,
    output_dir               : Optional[str | Path] = None,
    # --- detection ---
    tophat_radius            : int   = 25,
    gaussian_sigma           : float = 1.0,
    otsu_factor              : float = 0.8,
    min_object_area          : int   = 200,
    use_frangi               : bool  = False,
    # --- tracing ---
    min_fiber_length_px      : int   = 50,
    skeleton_smooth_window   : int   = 5,
    # --- measurement ---
    min_segment_length_um    : float = 0.5,
    smooth_profile_sigma     : float = 2.0,
    channel_thresholds       : Optional[List[float]] = None,
    # --- output ---
    save_measurements        : bool  = True,
    save_geojson             : bool  = True,
    save_summary             : bool  = True,
    geojson_indent           : Optional[int] = None,
    verbose                  : bool  = True,
) -> Dict:
    """Run the full DNA fiber analysis pipeline on a single TIFF.

    Parameters
    ----------
    tiff_path : str or Path
        Input image file (.tif / .tiff / OME-TIFF).
    channel_names : list[str], optional
        Human-readable labels for each channel in channel order.
        Default: ['ch0', 'ch1', …].
        Typical: ['CldU', 'IdU'] or ['Alexa594', 'Alexa488'].
    um_per_px : float, optional
        Physical pixel size in µm.  Overrides metadata if supplied.
    output_dir : str or Path, optional
        Directory for output files.  Defaults to the same directory as the
        input TIFF.  Created if it does not exist.
    tophat_radius : int
        Background subtraction radius (px).  Should be larger than the
        widest fiber and smaller than the background scale.
    gaussian_sigma : float
        Noise-reduction Gaussian σ (px).  0 = disabled.
    otsu_factor : float
        Multiply Otsu threshold by this factor (< 1 = more permissive).
    min_object_area : int
        Discard connected components smaller than this (px²).
    use_frangi : bool
        Apply Frangi vesselness filter before thresholding.  Slower but
        can improve detection when background is uneven.
    min_fiber_length_px : int
        Discard fiber paths shorter than this (pixels).
    skeleton_smooth_window : int
        Half-window for moving-average smoothing of skeleton paths.
    min_segment_length_um : float
        Discard / merge segments shorter than this (µm).
    smooth_profile_sigma : float
        Gaussian σ applied to sampled intensity profiles before
        channel thresholding.
    channel_thresholds : list[float], optional
        Manual per-channel thresholds in [0, 1].  Auto-estimated if None.
    save_measurements : bool
        Write segment-level CSV to output_dir.
    save_geojson : bool
        Write QuPath GeoJSON to output_dir.
    save_summary : bool
        Write fiber-level summary CSV to output_dir.
    geojson_indent : int, optional
        JSON indent (None = compact, 2 = pretty-printed).
    verbose : bool
        Print progress messages.

    Returns
    -------
    dict with keys:
      ``measurements``   – pd.DataFrame (segment level)
      ``summary``        – pd.DataFrame (fiber level)
      ``geojson``        – dict (GeoJSON FeatureCollection)
      ``paths``          – list of (N,2) int32 arrays (skeleton paths)
      ``meta``           – dict (image metadata)
      ``mask``           – (H,W) bool array (fiber mask)
      ``skeleton``       – (H,W) bool array
    """
    tiff_path = Path(tiff_path)
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"  [{time.time()-t0:5.1f}s]  {msg}")

    log(f"Loading  {tiff_path.name}")
    stack, meta = load_tiff(tiff_path, channel_names=channel_names,
                            um_per_px=um_per_px)
    um = meta["um_per_px"]
    ch = meta["channel_names"]

    log(f"  {meta['n_channels']} channels, {meta['height']}×{meta['width']} px, "
        f"{um:.4f} µm/px")

    # ---- Pre-process -------------------------------------------------------
    log("Pre-processing (background subtraction + smoothing)")
    clean = preprocess(stack,
                       tophat_radius    = tophat_radius,
                       gaussian_sigma   = gaussian_sigma,
                       subtract_background = True)

    # ---- Detect fiber mask -------------------------------------------------
    log("Building fiber mask")
    mask = make_fiber_mask(clean,
                           otsu_factor     = otsu_factor,
                           min_object_area = min_object_area,
                           use_frangi      = use_frangi)
    n_fiber_px = int(mask.sum())
    log(f"  Fiber mask: {n_fiber_px:,} foreground pixels")

    # ---- Skeletonise -------------------------------------------------------
    log("Skeletonising")
    skel = skeletonize_mask(mask)
    log(f"  Skeleton: {int(skel.sum()):,} pixels")

    # ---- Trace paths -------------------------------------------------------
    log("Tracing skeleton paths")
    paths = trace_skeleton(skel,
                           min_length_px  = min_fiber_length_px,
                           smooth_window  = skeleton_smooth_window)
    log(f"  Found {len(paths)} fibers")

    # ---- Measure -----------------------------------------------------------
    log("Measuring channel segments")
    df_segs = measure_fibers(paths, clean, ch, um,
                             channel_thresholds    = channel_thresholds,
                             min_segment_length_um = min_segment_length_um,
                             smooth_profile_sigma  = smooth_profile_sigma)
    log(f"  {len(df_segs)} segments across {df_segs['fiber_id'].nunique() if not df_segs.empty else 0} fibers")

    # ---- Classify ----------------------------------------------------------
    log("Classifying fiber patterns")
    df_segs = classify_fibers(df_segs, channel_names=ch)
    df_summary = summarise_by_class(df_segs)
    if not df_summary.empty:
        counts = df_summary["fiber_class"].value_counts().to_dict()
        log("  Class distribution: " +
            ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # ---- GeoJSON -----------------------------------------------------------
    log("Building QuPath GeoJSON")
    gj = build_geojson(paths, df_segs, ch,
                       include_fiber_backbones = True,
                       include_segment_lines   = True,
                       min_segment_length_um   = min_segment_length_um)
    log(f"  {len(gj['features'])} GeoJSON features")

    # ---- Save outputs ------------------------------------------------------
    if output_dir is None:
        output_dir = tiff_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = tiff_path.stem

    if save_measurements and not df_segs.empty:
        p = output_dir / f"{stem}_segments.csv"
        df_segs.to_csv(p, index=False)
        log(f"  Saved: {p.name}")

    if save_summary and not df_summary.empty:
        p = output_dir / f"{stem}_fiber_summary.csv"
        df_summary.to_csv(p, index=False)
        log(f"  Saved: {p.name}")

    if save_geojson:
        p = output_dir / f"{stem}_fibers.geojson"
        write_geojson(gj, p, indent=geojson_indent)
        log(f"  Saved: {p.name}")

    log(f"Done  ({time.time()-t0:.1f}s total)")

    return {
        "measurements": df_segs,
        "summary"     : df_summary,
        "geojson"     : gj,
        "paths"       : paths,
        "meta"        : meta,
        "mask"        : mask,
        "skeleton"    : skel,
    }


def run_batch(
    tiff_paths   : List[str | Path],
    output_dir   : str | Path,
    common_kwargs: Optional[Dict] = None,
    verbose      : bool = True,
) -> pd.DataFrame:
    """Run ``run_analysis`` on a list of TIFF files.

    Parameters
    ----------
    tiff_paths : list
        List of paths to TIFF files.
    output_dir : str or Path
        Root output directory; a sub-folder per TIFF is created.
    common_kwargs : dict, optional
        Keyword arguments passed to every ``run_analysis`` call
        (e.g. ``channel_names``, ``um_per_px``).
    verbose : bool
        Print per-file progress.

    Returns
    -------
    pd.DataFrame
        Concatenated fiber-level summary across all files, with a
        ``source_file`` column.
    """
    if common_kwargs is None:
        common_kwargs = {}

    all_summaries = []
    output_dir    = Path(output_dir)

    for i, tp in enumerate(tiff_paths):
        tp = Path(tp)
        if verbose:
            print(f"\n[{i+1}/{len(tiff_paths)}]  {tp.name}")

        per_file_out = output_dir / tp.stem
        try:
            res = run_analysis(
                tp,
                output_dir = per_file_out,
                verbose    = verbose,
                **common_kwargs,
            )
            summ = res["summary"].copy()
            summ["source_file"] = tp.name
            all_summaries.append(summ)
        except Exception as exc:
            print(f"  ERROR processing {tp.name}: {exc}")
            continue

    if not all_summaries:
        return pd.DataFrame()

    combined = pd.concat(all_summaries, ignore_index=True)
    combined.to_csv(output_dir / "batch_fiber_summary.csv", index=False)
    if verbose:
        print(f"\nBatch complete — {len(combined)} fibers across {len(all_summaries)} files.")
    return combined
