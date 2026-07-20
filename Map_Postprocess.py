"""Local post-processing of the wall-to-wall VM inference outputs (see Wall_to_Wall_Maps.ipynb
Steps 4-5): mosaic the predicted-class / confidence tiles that come back from GCS into one
raster per year, and visualize the class map.

Kept separate from Map_Export.py on purpose: that module is the Earth Engine side (AOI +
embedding export) and imports `ee`; these are local rasterio operations on the downloaded
tiles, so they carry different (and heavier) dependencies. Nothing here touches Earth Engine.
"""
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

# NODATA value written by vm_predict.py (outside both the 1-7 class range and 0-100 confidence
# range), and the Level-1 class palette. The tuned model predicts [1, 3, 4, 5, 6, 7]; there is
# no Ice/Snow (id 2), and glaciers are masked out, so id 2 should not appear in a valid map.
NODATA = 255
CLASS_COLORS = {
    1: '#3a7ca5',  # Water
    2: '#d9f0f7',  # Ice/Snow (not predicted; here for completeness)
    3: '#ab2b2b',  # Developed
    4: '#d2b48c',  # Barren/Sparse
    5: '#1b7837',  # Trees
    6: '#c2a55c',  # Shrubs
    7: '#a6d96a',  # Herbaceous
}


def mosaic_tiles(tifs, out_path, nodata=NODATA):
    """Mosaic grid-aligned, non-overlapping GeoTIFF tiles into one file, memory-safe.

    Streams each tile into its window of the output, one at a time, so a state-sized mosaic
    never loads fully into RAM (rasterio.merge would build the whole array). The AlphaEarth
    export tiles share one aligned grid and do not overlap, so windowed writes reconstruct the
    mosaic exactly. The output is first filled with `nodata` in row strips so any gap not
    covered by a tile reads as nodata rather than the default 0 (0 is a valid confidence value,
    so this matters for the confidence mosaic).

    Args:
        tifs: List of tile paths (all sharing one CRS, resolution, and aligned grid).
        out_path: Output GeoTIFF path.
        nodata: NODATA value for gaps and for the output profile.

    Returns:
        (out_path, (height, width)) of the written mosaic.
    """
    with rasterio.open(tifs[0]) as s0:
        res_x, res_y = s0.res
        crs, dtype, count = s0.crs, s0.dtypes[0], s0.count
    bounds = []
    for f in tifs:                               # metadata only -- cheap
        with rasterio.open(f) as s:
            bounds.append(s.bounds)
    left, top = min(b.left for b in bounds), max(b.top for b in bounds)
    right, bottom = max(b.right for b in bounds), min(b.bottom for b in bounds)
    width = int(round((right - left) / res_x))
    height = int(round((top - bottom) / res_y))
    transform = rasterio.transform.from_origin(left, top, res_x, res_y)
    profile = dict(driver='GTiff', height=height, width=width, count=count, dtype=dtype, crs=crs,
                   transform=transform, nodata=nodata, compress='deflate', predictor=2,
                   tiled=True, bigtiff='if_safer')
    with rasterio.open(out_path, 'w', **profile) as dst:
        for r in range(0, height, 4096):         # initialise the grid to nodata in row strips
            h = min(4096, height - r)
            dst.write(np.full((count, h, width), nodata, dtype=dtype), window=Window(0, r, width, h))
        for f in tifs:                           # paint each tile into its aligned window
            with rasterio.open(f) as s:
                col = int(round((s.bounds.left - left) / res_x))
                row = int(round((top - s.bounds.top) / res_y))
                dst.write(s.read(), window=Window(col, row, s.width, s.height))
    return out_path, (height, width)


def find_map(map_dir, year):
    """Path to the {year} class map in map_dir, or None.

    Prefers the clipped LULC_Class1_forest_{year}.tif (Step 4b), falling back to the
    _mosaic.tif from Step 4a.
    """
    for name in (f'LULC_Class1_forest_{year}.tif', f'LULC_Class1_forest_{year}_mosaic.tif'):
        p = os.path.join(map_dir, name)
        if os.path.exists(p):
            return p
    return None


def class_areas(path, nodata=NODATA):
    """Class histogram of a class raster -> {class_id: hectares}, computed block-by-block.

    Never loads the whole raster into RAM. Assumes an equal-area CRS (EPSG:5070), where every
    pixel is a true fixed area, so a plain pixel count times pixel-area is unbiased.
    """
    counts = np.zeros(256, dtype=np.int64)
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            counts += np.bincount(src.read(1, window=window).ravel(), minlength=256)
        px_ha = abs(src.transform.a * src.transform.e) / 1e4  # EPSG:5070 -> 0.01 ha/pixel at 10 m
    return {c: counts[c] * px_ha for c in range(256) if counts[c] and c != nodata}


def overview_rgb(path, max_dim=2000, class_colors=None):
    """Decimated (nearest-neighbour) read of a class raster, rendered to an RGB float image.

    Nearest-neighbour so categorical classes are not blurred; decimated so display of a
    state-sized raster stays light. Pixels with no colour (nodata, unmapped ids) render white.

    Args:
        path: Class raster path.
        max_dim: Target size of the longer side of the overview, in pixels.
        class_colors: {class_id: '#rrggbb'}; defaults to CLASS_COLORS.

    Returns:
        (H, W, 3) float array in [0, 1] for matplotlib imshow.
    """
    class_colors = class_colors or CLASS_COLORS
    with rasterio.open(path) as src:
        scale = max(1, round(max(src.width, src.height) / max_dim))
        arr = src.read(1, out_shape=(src.height // scale, src.width // scale),
                       resampling=Resampling.nearest)
    rgb = np.ones((*arr.shape, 3))               # white where nodata / no colour
    for cid, hexc in class_colors.items():
        m = arr == cid
        if m.any():
            rgb[m] = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return rgb
