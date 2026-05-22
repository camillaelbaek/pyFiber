"""
fiber_analysis — Python package for DNA fiber assay image analysis.

Pipeline overview
-----------------
1. io.py        – load single/multi-channel TIFF, extract pixel size from
                  OME-XML or user-supplied value
2. preprocess.py – per-channel background correction (white top-hat) +
                   Gaussian smoothing + optional contrast stretch
3. detect.py    – build a binary fiber mask from the combined channels,
                  morphological clean-up, skeletonisation
4. skeleton.py  – convert the skeleton image into ordered pixel paths via a
                  fast graph traversal; merge short dangling branches
5. measure.py   – for each path, sample per-channel signal, threshold to
                  assign channel labels per pixel, aggregate into segments
                  and convert pixel lengths → µm
6. classify.py  – classify each fiber by its segment pattern
                  (ongoing fork, new origin, terminated, bidirectional …)
7. qupath.py    – export a GeoJSON FeatureCollection that can be dragged
                  onto a QuPath project for immediate overlay
8. pipeline.py  – run_analysis() stitches everything together and returns a
                  results dict {measurements: DataFrame, geojson: dict}
"""

from .pipeline import run_analysis          # noqa: F401
from .io       import load_tiff             # noqa: F401
from .qupath   import write_geojson         # noqa: F401
