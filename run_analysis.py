#!/usr/bin/env python3
"""
run_analysis.py — Command-line entry point for fiber_analysis.

Usage examples
--------------
# Single file, auto-detect pixel size from OME metadata
python run_analysis.py slide01.tif --channels CldU IdU

# Supply pixel size manually and save pretty-printed GeoJSON
python run_analysis.py slide01.tif \
    --channels CldU IdU \
    --um-per-px 0.108 \
    --output-dir results/slide01 \
    --geojson-indent 2

# Batch: analyse every .tif in a folder
python run_analysis.py data/*.tif \
    --channels CldU IdU \
    --um-per-px 0.108 \
    --output-dir results/batch

# Tune detection sensitivity
python run_analysis.py slide01.tif \
    --channels CldU IdU \
    --um-per-px 0.108 \
    --otsu-factor 0.7 \
    --tophat-radius 30 \
    --use-frangi \
    --min-fiber-length-px 60
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import click

from fiber_analysis.pipeline import run_analysis, run_batch


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("tiff_paths", nargs=-1, required=True,
                type=click.Path(exists=True, dir_okay=False))
@click.option("--channels",         "-c",  multiple=True,
              default=("CldU", "IdU"), show_default=True,
              help="Channel names in order (repeat flag, e.g. -c CldU -c IdU).")
@click.option("--um-per-px",        "-u",  type=float, default=None,
              help="Physical pixel size in µm/px.  Overrides TIFF metadata.")
@click.option("--output-dir",       "-o",  type=click.Path(), default=None,
              help="Output directory.  Defaults to TIFF directory (single) or "
                   "current directory (batch).")
# Detection
@click.option("--tophat-radius",    type=int,   default=25,   show_default=True,
              help="White top-hat SE radius (px) for background subtraction.")
@click.option("--gaussian-sigma",   type=float, default=1.0,  show_default=True,
              help="Gaussian smoothing σ (px).  0 = disabled.")
@click.option("--otsu-factor",      type=float, default=0.8,  show_default=True,
              help="Multiply Otsu threshold by this (< 1 = more permissive).")
@click.option("--min-object-area",  type=int,   default=200,  show_default=True,
              help="Minimum fiber blob area (px²) — smaller blobs discarded.")
@click.option("--use-frangi",       is_flag=True, default=False,
              help="Apply Frangi vesselness filter before thresholding.")
# Tracing
@click.option("--min-fiber-length-px", type=int, default=50, show_default=True,
              help="Minimum skeleton path length in pixels.")
@click.option("--smooth-window",    type=int,   default=5,    show_default=True,
              help="Moving-average window for skeleton path smoothing.")
# Measurement
@click.option("--min-segment-um",   type=float, default=0.5,  show_default=True,
              help="Merge / discard segments shorter than this (µm).")
@click.option("--profile-sigma",    type=float, default=2.0,  show_default=True,
              help="Gaussian σ applied to intensity profiles before thresholding.")
# Output
@click.option("--geojson-indent",   type=int,   default=None,
              help="JSON indent level (omit for compact output).")
@click.option("--no-geojson",       is_flag=True, default=False,
              help="Skip GeoJSON output.")
@click.option("--no-csv",           is_flag=True, default=False,
              help="Skip CSV output.")
@click.option("--quiet", "-q",      is_flag=True, default=False,
              help="Suppress progress messages.")
def main(
    tiff_paths,
    channels,
    um_per_px,
    output_dir,
    tophat_radius,
    gaussian_sigma,
    otsu_factor,
    min_object_area,
    use_frangi,
    min_fiber_length_px,
    smooth_window,
    min_segment_um,
    profile_sigma,
    geojson_indent,
    no_geojson,
    no_csv,
    quiet,
):
    channel_names = list(channels) if channels else None
    verbose       = not quiet

    common = dict(
        channel_names          = channel_names,
        um_per_px              = um_per_px,
        tophat_radius          = tophat_radius,
        gaussian_sigma         = gaussian_sigma,
        otsu_factor            = otsu_factor,
        min_object_area        = min_object_area,
        use_frangi             = use_frangi,
        min_fiber_length_px    = min_fiber_length_px,
        skeleton_smooth_window = smooth_window,
        min_segment_length_um  = min_segment_um,
        smooth_profile_sigma   = profile_sigma,
        save_measurements      = not no_csv,
        save_summary           = not no_csv,
        save_geojson           = not no_geojson,
        geojson_indent         = geojson_indent,
        verbose                = verbose,
    )

    if len(tiff_paths) == 1:
        out = Path(output_dir) if output_dir else None
        run_analysis(tiff_paths[0], output_dir=out, **common)
    else:
        out = Path(output_dir) if output_dir else Path.cwd() / "fiber_results"
        run_batch(list(tiff_paths), output_dir=out,
                  common_kwargs=common, verbose=verbose)


if __name__ == "__main__":
    main()
