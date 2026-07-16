"""Earth Engine side of the wall-to-wall land-cover maps (see Wall_to_Wall_Maps.ipynb).

Builds the forest area of interest from RESOLVE Ecoregions 2017, and submits/polls the
masked AlphaEarth embedding exports to Cloud Storage. Inference itself does not happen
here: the exported tiles are scored on a Compute Engine VM in the bucket's region
(vm_predict.py), and only the small predicted-class tiles come back for mosaicking.
"""

import time

import ee
import pandas as pd

from Embedding_Utils import EMBEDDING_BANDS, EMBEDDING_COLLECTION_ID
from Geo_Utils import get_wa_or_geometry

ECOREGIONS_COLLECTION_ID = 'RESOLVE/ECOREGIONS/2017'

# Hand-picked ecoregions defining the mapped AOI within WA+OR. Transcribed from the dataset
# itself (see list_ecoregions()), not from memory -- ECO_NAME capitalisation is
# inconsistent ('Central-Southern Cascades Forests' vs '... forests'), and ee.Filter.inList
# matches exactly, so a guessed string would silently drop an entire ecoregion.
# Ecoregions extending outside WA/OR are trimmed by intersecting with the state boundary.
#
# This is every ecoregion in the 'Temperate Conifer Forests' biome that intersects the two
# states, PLUS the Willamette Valley oak savanna. The Willamette is a deliberate inclusion,
# not an oversight: RESOLVE classes it as 'Temperate Grasslands, Savannas & Shrublands'
# (it is historically oak savanna/prairie), so the AOI is not strictly a forest-biome
# selection. It adds ~14,900 km2 (~5% of the AOI).
FOREST_ECOREGIONS = [
    'Blue Mountains forests',
    'British Columbia coastal conifer forests',
    'Central Pacific Northwest coastal forests',
    'Central-Southern Cascades Forests',
    'Eastern Cascades forests',
    'Great Basin montane forests',
    'Klamath-Siskiyou forests',
    'North Cascades conifer forests',
    'Northern California coastal forests',
    'Northern Rockies conifer forests',
    'Puget lowland forests',
    'Willamette Valley oak savanna',  # grassland/savanna biome; included by explicit choice
]

# The other two ecoregions intersecting WA+OR, excluded as naturally non-forest. Recorded
# here so the include/exclude call is explicit rather than implied by absence.
EXCLUDED_ECOREGIONS = [
    'Snake-Columbia shrub steppe',  # Deserts & Xeric Shrublands
    'Palouse prairie',              # Temperate Grasslands, Savannas & Shrublands
]


def list_ecoregions(geometry=None):
    """Every RESOLVE ecoregion intersecting geometry, with its biome.

    The discovery step behind FOREST_ECOREGIONS: run this to read the exact ECO_NAME
    strings off the dataset before editing that list.

    Args:
        geometry: ee.Geometry to intersect; defaults to the dissolved WA+OR boundary.

    Returns:
        DataFrame with BIOME_NAME / ECO_NAME, sorted by biome, plus an 'included' flag
        showing whether each ecoregion is currently in FOREST_ECOREGIONS.
    """
    if geometry is None:
        geometry = get_wa_or_geometry()

    fc = ee.FeatureCollection(ECOREGIONS_COLLECTION_ID).filterBounds(geometry)
    info = fc.select(['ECO_NAME', 'BIOME_NAME'], retainGeometry=False).getInfo()

    rows = sorted({(f['properties']['BIOME_NAME'], f['properties']['ECO_NAME'])
                   for f in info['features']})
    df = pd.DataFrame(rows, columns=['BIOME_NAME', 'ECO_NAME'])
    df['included'] = df['ECO_NAME'].isin(FOREST_ECOREGIONS)
    return df


def get_forest_aoi(max_error=100):
    """The forest AOI: FOREST_ECOREGIONS clipped to the WA+OR boundary.

    Args:
        max_error: Maximum reprojection error in metres for dissolve/intersection.

    Returns:
        Tuple of (geometry, mask):
          geometry: ee.Geometry of forest ecoregions within WA+OR -- the Export region.
          mask:     ee.Image that is 1 over those ecoregions and masked elsewhere, so
                    updateMask() drops non-forest pixels rather than classifying them.
    """
    states = get_wa_or_geometry(max_error=max_error)
    forest_fc = (ee.FeatureCollection(ECOREGIONS_COLLECTION_ID)
                 .filter(ee.Filter.inList('ECO_NAME', FOREST_ECOREGIONS)))

    geometry = (forest_fc.geometry()
                .dissolve(maxError=max_error)
                .intersection(states, maxError=max_error))
    # Painting the FeatureCollection is cheaper than clipping to the dissolved polygon;
    # pixels outside the painted features stay masked, which is what updateMask wants.
    mask = ee.Image().byte().paint(forest_fc, 1).gt(0)
    return geometry, mask


def get_embedding_image(year, mask=None):
    """The AlphaEarth annual embedding mosaic for one year, forest-masked.

    Selects EMBEDDING_BANDS explicitly so band order matches the order the model was
    trained on (A00..A63) -- a silent reorder here would corrupt every prediction.

    Args:
        year: Calendar year (AlphaEarth is annual, 2017 onward).
        mask: Optional ee.Image mask (from get_forest_aoi) applied via updateMask.
    """
    image = (ee.ImageCollection(EMBEDDING_COLLECTION_ID)
             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
             .mosaic()
             .select(EMBEDDING_BANDS))
    if mask is not None:
        image = image.updateMask(mask)
    return image


def export_embeddings_to_gcs(year, geometry, mask, bucket, prefix='embeddings',
                             scale=10, crs='EPSG:5070', file_dimensions=1024,
                             max_pixels=1e13):
    """Submit a Cloud Storage export of one year's forest-masked embeddings.

    Args:
        year: Year to export.
        geometry, mask: From get_forest_aoi().
        bucket: GCS bucket name (no gs:// prefix).
        prefix: Object-name prefix; tiles land at {prefix}/{year}/embeddings_{year}*.tif.
        scale: Metres per pixel (10 = AlphaEarth's native resolution).
        crs: EPSG:5070 (NAD83 CONUS Albers) gives true 10 m pixels across both states,
            unlike EPSG:4326 which distorts at this latitude.
        file_dimensions: Tile size in pixels. 1024 keeps each tile ~268 MB uncompressed
            (1024*1024*64 bands*4 bytes); 2048 would exceed 1 GB per tile. Must be a
            multiple of the 256 px shard size.
        max_pixels: Export ceiling; the AOI is ~2.7e9 pixels, so 1e13 is ample.

    Returns:
        The started ee.batch.Task.
    """
    image = get_embedding_image(year, mask=mask)
    task = ee.batch.Export.image.toCloudStorage(
        image=image,
        description=f'alphaearth_embeddings_forest_{year}',
        bucket=bucket,
        fileNamePrefix=f'{prefix}/{year}/embeddings_{year}',
        region=geometry,
        scale=scale,
        crs=crs,
        fileDimensions=file_dimensions,
        maxPixels=max_pixels,
        fileFormat='GeoTIFF',
    )
    task.start()
    return task


def wait_for_tasks(tasks, poll_seconds=60, verbose=True):
    """Block until every task reaches a terminal state, printing status as it goes.

    Args:
        tasks: Iterable of started ee.batch.Task.
        poll_seconds: Seconds between status checks.
        verbose: Print a status line on each poll.

    Returns:
        Dict of task description -> final state. Raises nothing on FAILED: the caller
        should inspect the returned states (a failed export is reported, not swallowed).
    """
    tasks = list(tasks)
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED'}
    states = {}

    while True:
        states = {}
        for t in tasks:
            status = t.status()
            states[status.get('description', t.id)] = status['state']
        if verbose:
            summary = ', '.join(f'{d}: {s}' for d, s in states.items())
            print(f'[{time.strftime("%H:%M:%S")}] {summary}')
        if all(s in terminal for s in states.values()):
            return states
        time.sleep(poll_seconds)
