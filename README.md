# fiber_analysis

Python pipeline for **DNA fiber assay** image analysis.

Reads multi-channel microscopy TIFF files, detects and traces replication fibers,
measures segment lengths per fluorescent channel, classifies replication patterns,
and exports **GeoJSON overlays** for direct drag-and-drop visualization in **QuPath**.

Inspired by [FiberQ](https://github.com/pierreghesqui/FiberQ) but written entirely
in Python and operating directly on TIFF files without MATLAB or ImageJ.

---

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+.

---

## Quick start

### Single image

```bash
python run_analysis.py slide01.tif \
    --channels CldU IdU \
    --um-per-px 0.108 \
    --output-dir results/slide01
```

Output in `results/slide01/`:
```
slide01_segments.csv          # one row per fiber segment
slide01_fiber_summary.csv     # one row per fiber (class, lengths, ratios)
slide01_fibers.geojson        # drag onto QuPath for overlay
```

### Batch processing

```bash
python run_analysis.py data/*.tif \
    --channels CldU IdU \
    --um-per-px 0.108 \
    --output-dir results/batch
```

Creates one sub-folder per TIFF plus a combined `batch_fiber_summary.csv`.

### QC figure (tune parameters)

```bash
python visualize_qc.py slide01.tif \
    --channels CldU IdU \
    --um-per-px 0.108
```

Produces `slide01_qc.png` with 8 panels showing each pipeline step.

### Python API

```python
from fiber_analysis import run_analysis

results = run_analysis(
    tiff_path     = "slide01.tif",
    channel_names = ["CldU", "IdU"],
    um_per_px     = 0.108,
    output_dir    = "results/slide01",
)

df   = results["measurements"]   # segment-level DataFrame
summ = results["summary"]         # fiber-level DataFrame
gj   = results["geojson"]         # GeoJSON dict
```

---

## QuPath overlay

1. Open your TIFF in QuPath (drag file onto project or use *File → Open*).
2. Make sure the image entry is selected.
3. Drag `<stem>_fibers.geojson` onto the QuPath viewer **or**
   use *File → Import → Import objects from GeoJSON*.
4. Fiber backbones appear in grey; channel segments are coloured:
   - **CldU** → red (`#D62728`)
   - **IdU** → green (`#2CA02C`)
5. Click any annotation to see its measurements in the *Annotations* panel.

> **Tip:** QuPath pixel coordinates (0,0) = top-left, which matches NumPy
> image convention.  The GeoJSON uses `[col, row]` as `[x, y]` — exactly
> what QuPath expects.

---

## Input TIFF requirements

| Requirement | Notes |
|---|---|
| Format | `.tif` / `.tiff` / OME-TIFF |
| Channels | 2 (typical: CldU + IdU) or more |
| Bit depth | 8, 12, 16, 32 bit (auto-scaled to [0,1]) |
| Pixel size | Read from OME-XML; supply `--um-per-px` if absent |
| Channel order | Positional: first channel = first pulse label |
| Z-stacks | Max-projection applied automatically |

---

## Key parameters

### Detection (`make_fiber_mask`)

| Parameter | Default | Effect |
|---|---|---|
| `tophat_radius` | 25 px | Background subtraction scale; increase for uneven backgrounds |
| `otsu_factor` | 0.8 | < 1 = more signal kept; raise if spurious blobs appear |
| `min_object_area` | 200 px² | Discard small blobs (nuclei bleed-through, dust) |
| `use_frangi` | False | Vesselness filter; helps when background is very uneven |

### Tracing (`trace_skeleton`)

| Parameter | Default | Effect |
|---|---|---|
| `min_fiber_length_px` | 50 px | Discard short skeleton fragments |
| `smooth_window` | 5 px | Path smoothing; reduce staircase artefacts |

### Measurement (`measure_fibers`)

| Parameter | Default | Effect |
|---|---|---|
| `min_segment_length_um` | 0.5 µm | Merge/discard label-flicker segments |
| `smooth_profile_sigma` | 2.0 px | Gaussian σ on intensity profile before thresholding |

---

## Output columns

### `_segments.csv`

| Column | Description |
|---|---|
| `fiber_id` | Zero-based fiber index |
| `segment_id` | Segment index within fiber |
| `channel` | Channel label (e.g. `CldU`, `IdU`, `unlabelled`) |
| `length_px` | Segment length in pixels (Euclidean) |
| `length_um` | Segment length in µm |
| `start_row`, `start_col` | Start pixel coordinates |
| `end_row`, `end_col` | End pixel coordinates |
| `fiber_total_length_um` | Total fiber length (µm) |
| `fiber_class` | Classification (see below) |
| `segment_pattern` | Compact pattern string e.g. `CldU→IdU` |
| `ratio_2nd_1st` | Length(IdU) / Length(CldU) per fiber |

### `_fiber_summary.csv`

One row per fiber, with per-channel length sums added as `length_um_CldU` etc.

---

## Fiber classification scheme

| Class | Pattern | Biology |
|---|---|---|
| `ongoing_fork` | ch0 → ch1 (or ch1 → ch0) | Fork active through both pulses |
| `stalled_first_pulse` | ch0 only | Fork stalled before second pulse |
| `new_origin` | ch1 only | Origin fired after first pulse |
| `bidirectional_old` | ch0 → ch1 → ch0 | Bi-directional fork from old origin |
| `bidirectional_new` | ch1 → ch0 → ch1 | Bi-directional fork from new origin |
| `restart` | ch0 → ch1 → ch0 → … | Stalled then restarted |
| `complex` | Any other multi-segment pattern | Inspect manually |
| `unlabelled` | No labeled segments | Artifact or debris |

Default: ch0 = CldU (first pulse), ch1 = IdU (second pulse).

---

## Package structure

```
fiber_analysis/
├── __init__.py      exports run_analysis, load_tiff, write_geojson
├── io.py            TIFF loading + OME pixel-size extraction
├── preprocess.py    background subtraction + smoothing
├── detect.py        fiber mask + skeletonisation
├── skeleton.py      graph-based skeleton → ordered paths
├── measure.py       per-channel signal sampling + segment measurement
├── classify.py      pattern classification + fiber summary
├── qupath.py        GeoJSON builder (QuPath-compatible)
└── pipeline.py      run_analysis() + run_batch()

run_analysis.py      CLI entry point (uses Click)
visualize_qc.py      8-panel QC figure for parameter tuning
requirements.txt
```

---

## Tips

* **Start with the QC figure** to tune `otsu_factor` and `tophat_radius`
  before running a batch.
* For images with very sparse fibers, try `otsu_factor=0.6–0.7`.
* For dense images with overlapping fibers, `use_frangi=True` can help.
* The `min_segment_length_um` parameter is critical for clean classification;
  values of 0.5–2 µm work well for typical labeling pulses.
* In QuPath, use *Measure → Show measurement table* after importing GeoJSON
  to export all fiber measurements to a spreadsheet.
