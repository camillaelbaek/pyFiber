"""
fiber_analysis.classify
-----------------------
Classify each fiber by the ordered sequence of its channel segments.

Labeling convention assumed
---------------------------
  ch_names[0]  =  first-pulse label  (e.g. CldU / Alexa594 / red)
  ch_names[1]  =  second-pulse label (e.g. IdU  / Alexa488 / green)

Any "unlabelled" segment is treated as a gap.

Classification categories
-------------------------
Category                Description
----------------------  ------------------------------------------------
ongoing_fork            Exactly two labeled segments in sequence,
                        one of each pulse (either order).
                        Unidirectional active fork through both pulses.
new_origin              Single second-pulse (IdU only) segment.
                        Fork fired after the first pulse.
stalled_first_pulse     Single first-pulse (CldU only) segment.
                        Fork stalled before the second pulse.
bidirectional_old       Three segments: 1st–2nd–1st pulse (R-G-R).
                        Origin fired before the first pulse; two forks.
bidirectional_new       Three segments: 2nd–1st–2nd pulse (G-R-G).
                        Origin fired after the first pulse.
                        (rare; may indicate complex rearrangement)
restart                 Two or more alternating segments of the same
                        first-pulse label separated by a second-pulse
                        segment (stalled then restarted).
complex                 Any other multi-segment pattern.
unlabelled              No labeled segments found.

For experiments with > 2 channels, the classification defaults to
'complex' for patterns that do not fit the two-pulse scheme; the raw
segment sequence is still reported in the 'segment_pattern' column.

Returned columns (added to the input DataFrame)
-----------------------------------------------
fiber_class     : str   — one of the categories above
segment_pattern : str   — compact string e.g. "CldU→IdU→CldU"
ratio_2nd_1st   : float — length(2nd pulse) / length(1st pulse) per fiber
                          (NaN for fibers where either pulse is absent)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_fibers(
    df: pd.DataFrame,
    channel_names: Optional[List[str]] = None,
    gap_label: str = "unlabelled",
) -> pd.DataFrame:
    """Add classification columns to the segment DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output from ``measure_fibers`` — one row per segment.
    channel_names : list[str], optional
        If supplied and has exactly 2 entries, the first is treated as
        the first-pulse label and the second as the second-pulse label.
        Otherwise, classification is limited to 'complex' / 'unlabelled'.
    gap_label : str
        Label used for unlabelled pixels (default: 'unlabelled').

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns:
        ``fiber_class``, ``segment_pattern``, ``ratio_2nd_1st``.
    """
    if df.empty:
        df = df.copy()
        df["fiber_class"]     = pd.NA
        df["segment_pattern"] = pd.NA
        df["ratio_2nd_1st"]   = np.nan
        return df

    pulse1 = channel_names[0] if (channel_names and len(channel_names) >= 1) else None
    pulse2 = channel_names[1] if (channel_names and len(channel_names) >= 2) else None

    result_rows = []
    for fid, grp in df.groupby("fiber_id", sort=True):
        grp_sorted = grp.sort_values("segment_id")
        labels     = grp_sorted["channel"].tolist()

        # Compact sequence: remove gap labels for classification logic
        active = [l for l in labels if l != gap_label]

        pattern = "→".join(labels)
        cls, ratio = _classify_sequence(active, pulse1, pulse2)

        for _, row in grp_sorted.iterrows():
            result_rows.append({
                **row.to_dict(),
                "fiber_class"    : cls,
                "segment_pattern": pattern,
                "ratio_2nd_1st"  : ratio,
            })

    out = pd.DataFrame(result_rows).reset_index(drop=True)
    return out


def summarise_by_class(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a fiber-level summary (one row per fiber) with class info.

    Useful for quick population counts and fork-rate calculations.
    """
    if df.empty:
        return df

    records = []
    for fid, grp in df.groupby("fiber_id", sort=True):
        first_row = grp.iloc[0]
        p1_len = grp.loc[grp["channel"] == grp["channel"].iloc[0], "length_um"].sum()

        # Lengths per channel
        ch_lens = grp.groupby("channel")["length_um"].sum().to_dict()

        records.append({
            "fiber_id"             : fid,
            "fiber_class"          : first_row.get("fiber_class"),
            "segment_pattern"      : first_row.get("segment_pattern"),
            "n_segments"           : int(first_row.get("n_segments", len(grp))),
            "fiber_total_length_um": float(first_row.get("fiber_total_length_um", np.nan)),
            "ratio_2nd_1st"        : first_row.get("ratio_2nd_1st"),
            **{f"length_um_{k}": v for k, v in ch_lens.items()},
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_sequence(
    active : List[str],
    pulse1 : Optional[str],
    pulse2 : Optional[str],
) -> tuple:
    """Return (fiber_class_str, ratio_float)."""

    if not active:
        return "unlabelled", np.nan

    if pulse1 is None or pulse2 is None:
        # Cannot apply two-pulse classification
        return "complex", np.nan

    p1, p2 = pulse1, pulse2

    # Deduplicate consecutive identical labels
    dedup = [active[0]]
    for lbl in active[1:]:
        if lbl != dedup[-1]:
            dedup.append(lbl)

    ratio = np.nan

    # Single segment
    if len(dedup) == 1:
        if dedup[0] == p1:
            return "stalled_first_pulse", np.nan
        if dedup[0] == p2:
            return "new_origin", np.nan
        return "complex", np.nan

    # Two segments
    if len(dedup) == 2:
        if set(dedup) == {p1, p2}:
            return "ongoing_fork", np.nan
        return "complex", np.nan

    # Three segments
    if len(dedup) == 3:
        if dedup[0] == p1 and dedup[1] == p2 and dedup[2] == p1:
            return "bidirectional_old", np.nan
        if dedup[0] == p2 and dedup[1] == p1 and dedup[2] == p2:
            return "bidirectional_new", np.nan

    # Check restart: alternating p1 ... p2 ... p1
    if len(dedup) >= 4 and dedup[0] == p1 and dedup[-1] == p1:
        return "restart", np.nan

    return "complex", np.nan
