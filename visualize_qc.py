"""
visualize_qc.py — Quick QC plots for a single TIFF.

Produces a multi-panel figure showing:
  Panel 1  Raw max-projection (all channels merged)
  Panel 2  Pre-processed max-projection
  Panel 3  Fiber mask (binary)
  Panel 4  Skeleton overlaid on raw image
  Panel 5  Traced fiber paths coloured by fiber_id
  Panel 6  Segments coloured by channel assignment
  Panel 7  Histogram of fiber lengths (µm)
  Panel 8  Class distribution bar chart

Usage
-----
python visualize_qc.py slide01.tif --channels CldU IdU --um-per-px 0.108
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")      # headless rendering
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import click

from fiber_analysis.io        import load_tiff
from fiber_analysis.preprocess import preprocess
from fiber_analysis.detect    import make_fiber_mask, skeletonize_mask
from fiber_analysis.skeleton  import trace_skeleton
from fiber_analysis.measure   import measure_fibers
from fiber_analysis.classify  import classify_fibers, summarise_by_class


# Channel-to-color mapping for segment overlay
_SEG_COLORS = {
    "CldU" : "#D62728",
    "IdU"  : "#2CA02C",
    "ch0"  : "#D62728",
    "ch1"  : "#2CA02C",
    "unlabelled": "#AAAAAA",
}


@click.command()
@click.argument("tiff_path", type=click.Path(exists=True))
@click.option("--channels",        "-c",  multiple=True, default=("CldU","IdU"))
@click.option("--um-per-px",       "-u",  type=float, default=None)
@click.option("--output",          "-o",  type=click.Path(), default=None,
              help="Output PNG path.  Default: <stem>_qc.png next to the TIFF.")
@click.option("--otsu-factor",     type=float, default=0.8)
@click.option("--tophat-radius",   type=int,   default=25)
@click.option("--min-fiber-px",    type=int,   default=50)
@click.option("--show",            is_flag=True, default=False,
              help="Open the figure interactively (requires a display).")
def main(tiff_path, channels, um_per_px, output, otsu_factor,
         tophat_radius, min_fiber_px, show):

    tiff_path     = Path(tiff_path)
    channel_names = list(channels)

    print(f"Loading {tiff_path.name} …")
    stack, meta = load_tiff(tiff_path, channel_names=channel_names,
                            um_per_px=um_per_px)
    um = meta["um_per_px"]

    print("Pre-processing …")
    clean = preprocess(stack, tophat_radius=tophat_radius)

    print("Detecting fibers …")
    mask = make_fiber_mask(clean, otsu_factor=otsu_factor)
    skel = skeletonize_mask(mask)

    print("Tracing paths …")
    paths = trace_skeleton(skel, min_length_px=min_fiber_px)

    print("Measuring …")
    df_segs    = measure_fibers(paths, clean, channel_names, um)
    df_segs    = classify_fibers(df_segs, channel_names=channel_names)
    df_summary = summarise_by_class(df_segs)

    # ------------------------------------------------------------------ #
    # Compose figure                                                       #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    ax = axes.ravel()
    _title_kw = dict(fontsize=9, fontweight="bold")

    # 1. Raw composite
    rgb_raw = _to_rgb(stack, channel_names)
    ax[0].imshow(rgb_raw)
    ax[0].set_title("Raw (max proj.)", **_title_kw)
    ax[0].axis("off")

    # 2. Pre-processed composite
    rgb_clean = _to_rgb(clean, channel_names)
    ax[1].imshow(rgb_clean)
    ax[1].set_title("Pre-processed", **_title_kw)
    ax[1].axis("off")

    # 3. Fiber mask
    ax[2].imshow(mask, cmap="gray")
    ax[2].set_title(f"Fiber mask  ({int(mask.sum()):,} px)", **_title_kw)
    ax[2].axis("off")

    # 4. Skeleton overlay
    ax[3].imshow(rgb_raw)
    skel_rows, skel_cols = np.where(skel)
    ax[3].scatter(skel_cols, skel_rows, s=0.2, c="yellow", linewidths=0)
    ax[3].set_title(f"Skeleton  ({len(paths)} fibers)", **_title_kw)
    ax[3].axis("off")

    # 5. Traced paths coloured by fiber_id
    ax[4].imshow(rgb_raw)
    cmap_fibers = plt.cm.get_cmap("tab20", max(len(paths), 1))
    for fid, path in enumerate(paths):
        ax[4].plot(path[:, 1], path[:, 0],
                   lw=1.2, color=cmap_fibers(fid % 20), alpha=0.85)
    ax[4].set_title("Traced paths", **_title_kw)
    ax[4].axis("off")

    # 6. Segment overlay coloured by channel
    ax[5].imshow(rgb_raw)
    if not df_segs.empty:
        for _, seg in df_segs.iterrows():
            fid = int(seg["fiber_id"])
            if fid >= len(paths):
                continue
            path = paths[fid]
            # find sub-path indices
            from fiber_analysis.qupath import _slice_path  # noqa
            sub = _slice_path(path,
                              int(seg["start_row"]), int(seg["start_col"]),
                              int(seg["end_row"]),   int(seg["end_col"]))
            col = _SEG_COLORS.get(seg["channel"], "#FF00FF")
            ax[5].plot(sub[:, 1], sub[:, 0], lw=2.5, color=col, alpha=0.9)
    ax[5].set_title("Channel segments", **_title_kw)
    ax[5].axis("off")
    # legend
    for cname, ccol in _SEG_COLORS.items():
        if cname in channel_names or cname == "unlabelled":
            ax[5].plot([], [], color=ccol, lw=2.5, label=cname)
    ax[5].legend(fontsize=7, loc="lower right")

    # 7. Fiber length histogram
    if not df_summary.empty:
        lengths = df_summary["fiber_total_length_um"].dropna()
        ax[6].hist(lengths, bins=30, color="#4878CF", edgecolor="white", linewidth=0.5)
        ax[6].set_xlabel("Fiber length (µm)", fontsize=8)
        ax[6].set_ylabel("Count", fontsize=8)
        ax[6].set_title(f"Fiber lengths  (n={len(lengths)})", **_title_kw)
        ax[6].tick_params(labelsize=7)
    else:
        ax[6].text(0.5, 0.5, "No fibers detected", ha="center", va="center",
                   transform=ax[6].transAxes, fontsize=10, color="gray")
        ax[6].set_title("Fiber lengths", **_title_kw)
        ax[6].axis("off")

    # 8. Class distribution
    if not df_summary.empty and "fiber_class" in df_summary.columns:
        counts = df_summary["fiber_class"].value_counts()
        colors = plt.cm.tab10.colors[:len(counts)]
        ax[7].barh(counts.index, counts.values,
                   color=colors, edgecolor="white", linewidth=0.5)
        ax[7].set_xlabel("Count", fontsize=8)
        ax[7].set_title("Fiber class distribution", **_title_kw)
        ax[7].tick_params(labelsize=7)
    else:
        ax[7].text(0.5, 0.5, "No classifications", ha="center", va="center",
                   transform=ax[7].transAxes, fontsize=10, color="gray")
        ax[7].set_title("Class distribution", **_title_kw)
        ax[7].axis("off")

    fig.suptitle(
        f"{tiff_path.name}  |  {meta['n_channels']} ch  |  {um:.4f} µm/px  |  "
        f"{len(paths)} fibers",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------ #
    # Save / show                                                          #
    # ------------------------------------------------------------------ #
    if output is None:
        output = tiff_path.parent / f"{tiff_path.stem}_qc.png"
    output = Path(output)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"QC figure saved: {output}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_rgb(stack: np.ndarray, channel_names: List[str]) -> np.ndarray:
    """Map 1-3 channels to a displayable (H, W, 3) uint8 RGB image."""
    H, W = stack.shape[1], stack.shape[2]
    rgb  = np.zeros((H, W, 3), dtype=np.float32)

    ch_to_rgb = {}
    for i, name in enumerate(channel_names):
        name_lower = name.lower()
        if "cldu" in name_lower or "red" in name_lower or name_lower in ("ch0",):
            ch_to_rgb[i] = 0   # red
        elif "idu" in name_lower or "green" in name_lower or name_lower in ("ch1",):
            ch_to_rgb[i] = 1   # green
        elif "blue" in name_lower or "dapi" in name_lower or name_lower in ("ch2",):
            ch_to_rgb[i] = 2   # blue
        else:
            ch_to_rgb[i] = i % 3

    for c in range(stack.shape[0]):
        ch_plane = stack[c]
        ax = ch_to_rgb.get(c, 0)
        rgb[:, :, ax] = np.maximum(rgb[:, :, ax], ch_plane)

    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


if __name__ == "__main__":
    main()
