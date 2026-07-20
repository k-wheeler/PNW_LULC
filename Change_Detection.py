"""CCDC change-dating on Landsat, for the change-detection step (see the plan's
"Change detection" section and Pilot_LaborDayFires.ipynb).

CCDC (Continuous Change Detection and Classification, Zhu & Woodcock 2014) fits harmonic
time-series models to dense Landsat surface reflectance and flags a break when the residual
pattern shifts persistently. It runs on Landsat, NOT AlphaEarth: CCDC needs many intra-annual
observations to fit the seasonal model, and AlphaEarth is one vector per year.

Two things to know before using this:
  * CCDC is expensive. Over a whole AOI it exceeds the interactive memory limit -- run it
    masked to candidate pixels, or sample points, or export server-side. The pilot samples
    points inside the burn.
  * tBreak is an ARRAY per pixel (multiple breakpoints possible), in the units set by
    `dateFormat`. We use dateFormat=1 (fractional years), so a Sept-8-2020 break reads as
    ~2020.688. fire_window_break() pulls out the one break inside a given year window.
"""

import datetime as dt

import ee
import pandas as pd

# Harmonised band names CCDC's breakpoint/tmask bands refer to.
BANDS = ['BLUE', 'GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2']

# Hansen Global Forest Change -- pre-computed annual forest-loss year, an independent
# cross-check on CCDC break dates. v1_13 is current; v1_12 and earlier are DEPRECATED.
HANSEN_ID = 'UMD/hansen/global_forest_change_2025_v1_13'

# Landsat Collection-2 Level-2 (surface reflectance) source bands, per sensor. L8/L9 share a
# band layout; L5/L7 share a different one (SR_B1..B5,B7 vs SR_B2..B7).
_SR_BANDS = {
    'LANDSAT/LC09/C02/T1_L2': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    'LANDSAT/LC08/C02/T1_L2': ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    'LANDSAT/LE07/C02/T1_L2': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
    'LANDSAT/LT05/C02/T1_L2': ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
}

# QA_PIXEL bits 1-4 = dilated cloud, cirrus, cloud, cloud shadow. Require all clear.
_QA_CLEAR_BITMASK = int('11110', 2)


def _prep_sensor(coll_id):
    """One sensor's C02 L2 collection: harmonised band names, scaled to reflectance, cloud-masked."""
    sr_bands = _SR_BANDS[coll_id]

    def _f(img):
        clear = img.select('QA_PIXEL').bitwiseAnd(_QA_CLEAR_BITMASK).eq(0)
        reflectance = img.select(sr_bands, BANDS).multiply(0.0000275).add(-0.2)
        return reflectance.updateMask(clear).copyProperties(img, ['system:time_start'])

    return ee.ImageCollection(coll_id).map(_f)


def prepare_landsat(region, start, end,
                    sensors=('LANDSAT/LE07/C02/T1_L2',
                             'LANDSAT/LC08/C02/T1_L2',
                             'LANDSAT/LC09/C02/T1_L2')):
    """Harmonised, cloud-masked Landsat SR collection over region/time for CCDC input.

    Args:
        region: ee.Geometry to filter scenes to. Keep this small (buffered sample points,
            or the candidate-change mask) -- CCDC over a large area blows the memory limit.
        start, end: 'YYYY-MM-DD' date bounds. A window that brackets the event with a couple
            of years on each side gives CCDC enough observations to fit pre- and post-break
            segments; too short a post-break tail and it cannot confirm the break.
        sensors: Collection IDs to merge. Default L7+L8+L9 (adding L7 raises temporal
            density, which improves detection despite its SLC-off gaps). Add LT05 for
            pre-2012 work.

    Returns:
        An ee.ImageCollection with bands BANDS and system:time_start.
    """
    coll = None
    for cid in sensors:
        c = _prep_sensor(cid)
        coll = c if coll is None else coll.merge(c)
    return coll.filterBounds(region).filterDate(start, end)


def add_indices(image):
    """Add NDVI and NBR (burn index) bands, computed from the harmonised BANDS.

    NDVI = (NIR-RED)/(NIR+RED) -- vegetation vigor; drops when canopy is lost.
    NBR  = (NIR-SWIR2)/(NIR+SWIR2) -- burn index; drops sharply right after fire and recovers
        slowly as vegetation regrows. The standard index for visualising/dating burns.
    """
    ndvi = image.normalizedDifference(['NIR', 'RED']).rename('NDVI')
    nbr = image.normalizedDifference(['NIR', 'SWIR2']).rename('NBR')
    return image.addBands([ndvi, nbr])


def point_index_timeseries(collection, lon, lat, scale=30):
    """NDVI/NBR time series at one point, as a DataFrame (date, NDVI, NBR).

    Uses ImageCollection.getRegion on a SINGLE point geometry. This is deliberately not
    reduceRegions on a FeatureCollection of many points at once: reduceRegions' output does
    NOT reliably preserve input order (confirmed the hard way -- see Pilot_LaborDayFires.ipynb,
    where an order-preservation assumption silently broke going from 50 to 60 points). A
    single-point query has no such ambiguity: every returned row unambiguously belongs to
    this point, at the cost of one Earth Engine call per point rather than one batched call.

    Args:
        collection: An ee.ImageCollection with NDVI/NBR bands (see add_indices) and
            system:time_start.
        lon, lat: Point coordinates (EPSG:4326).
        scale: Sample scale in metres.

    Returns:
        DataFrame with columns date, NDVI, NBR, sorted by date, with cloud-masked (null)
        observations dropped.
    """
    pt = ee.Geometry.Point([lon, lat])
    rows = collection.select(['NDVI', 'NBR']).getRegion(pt, scale=scale).getInfo()
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df['date'] = pd.to_datetime(df['time'], unit='ms')
    df = df.dropna(subset=['NDVI', 'NBR']).sort_values('date').reset_index(drop=True)
    return df[['date', 'NDVI', 'NBR']]


def run_ccdc(collection, breakpoint_bands=('GREEN', 'RED', 'NIR', 'SWIR1', 'SWIR2'),
             tmask_bands=('GREEN', 'SWIR2'), min_observations=6,
             chi_square_probability=0.99, min_num_of_years_scaler=1.33,
             lambda_=20, max_iterations=10000):
    """Run CCDC with fractional-year break dates (dateFormat=1).

    Defaults follow the common CCDC configuration (Arévalo et al. / GEE examples). Returns
    the raw CCDC result image; its tBreak band is a per-pixel array of break dates in
    fractional years. `lambda` is a Python keyword, hence the `lambda_` argument and the
    dict-splat below.
    """
    return ee.Algorithms.TemporalSegmentation.Ccdc(**{
        'collection': collection,
        'breakpointBands': list(breakpoint_bands),
        'tmaskBands': list(tmask_bands),
        'minObservations': min_observations,
        'chiSquareProbability': chi_square_probability,
        'minNumOfYearsScaler': min_num_of_years_scaler,
        'dateFormat': 1,          # fractional years
        'lambda': lambda_,
        'maxIterations': max_iterations,
    })


def fire_window_break(ccdc_image, start_year, end_year):
    """Scalar image of the CCDC break date (fractional years) inside [start_year, end_year).

    Pulls the one break in the window out of the per-pixel tBreak array. Pixels with no break
    in the window get 0 (arrayPad backfills the empty array), so mask with `.gt(0)` before use.

    Args:
        ccdc_image: Output of run_ccdc().
        start_year, end_year: Integer year bounds; a 2020 fire uses (2020, 2021).

    Returns:
        Single-band ee.Image 'break_year' (fractional years, 0 = no break in window).
    """
    tbreak = ee.Image(ccdc_image).select('tBreak')
    in_window = tbreak.gte(start_year).And(tbreak.lt(end_year))
    return (tbreak.arrayMask(in_window)
            .arrayPad([1])
            .arrayGet([0])
            .rename('break_year'))


def frac_year_to_month(frac_year):
    """Fractional year -> 1-indexed month (e.g. 2020.688 -> ~9.3, September).

    Plain arithmetic for interpreting a scalar break date. CCDC break dates are only as sharp
    as Landsat observation density, so treat sub-month precision cautiously -- especially for
    Nov-Mar breaks, which the PNW cloud season leaves poorly constrained.
    """
    return (frac_year - int(frac_year)) * 12 + 1


def frac_year_to_date(frac_year):
    """Fractional year -> calendar date (e.g. 2020.688 -> date(2020, 9, 8)).

    CCDC's dateFormat=1 uses a 1-indexed day-of-year/days-in-year convention (day 1 = Jan 1),
    verified against real CCDC output: the Holiday Farm Fire's known 2020-09-08 ignition
    (day 252 of a 366-day leap year, 252/366 = 2020.6885) matches CCDC's actual break values
    (median 2020.688) to the day. Respects leap years, unlike frac_year_to_month's coarser
    30-day-month approximation. Use this for plotting/labelling actual dates; use
    frac_year_to_month for quick month-level summaries (e.g. the break-month histogram).

    Treat this as accurate to about a day, not sub-day -- CCDC's own precision is bounded by
    Landsat's ~16-day revisit interval.
    """
    year = int(frac_year)
    days_in_year = (dt.date(year + 1, 1, 1) - dt.date(year, 1, 1)).days  # 365 or 366
    # max(1, ...): at frac_year exactly X.0, the raw round() gives day 0, which would wrap to
    # Dec 31 of year X-1 instead of Jan 1 of year X.
    day_of_year_1indexed = max(1, round((frac_year - year) * days_in_year))
    return dt.date(year, 1, 1) + dt.timedelta(days=day_of_year_1indexed - 1)


def plot_index_timeseries(ax, i, row, ts, extra_label=None):
    """Plot one point's NDVI/NBR time series on `ax`, with fire ignition + CCDC break marked.

    Shared by the pilot's PDF-writing extraction loop and its standalone inline preview (which
    reads cached data back rather than re-plotting from a live query), so both render
    identically. `row` needs 'break_year', 'sample_type', 'severity_name', 'to_class_name'
    (see Pilot_LaborDayFires.ipynb's ccdc_breaks); `ts` needs 'date', 'NDVI', 'NBR' (see
    point_index_timeseries).
    """
    ax.plot(ts['date'], ts['NDVI'], 'o-', color='#2e7d32', ms=3, lw=1, label='NDVI')
    ax.plot(ts['date'], ts['NBR'], 'o-', color='#c44e52', ms=3, lw=1, label='NBR (burn index)')
    ax.axvline(pd.Timestamp('2020-09-08'), color='k', ls=':', alpha=0.5, label='fire ignition')
    if pd.notna(row['break_year']):
        bd = frac_year_to_date(row['break_year'])
        ax.axvline(pd.Timestamp(bd), color='b', ls='--', label=f'CCDC break ({bd})')
        break_note = f'break {bd}'
    else:
        break_note = 'no break detected'
    ax.set_ylim(-0.4, 1.05)
    ax.set_xlabel('date'); ax.set_ylabel('index value')
    title = (f"Point {i:02d} ({row['sample_type']}, {row['severity_name']}, "
            f"to {row['to_class_name']}), {break_note}")
    if extra_label:
        title = f'{extra_label}\n{title}'
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc='lower left')


# Points featured in the pilot's inline preview: first match per label wins. Pilot-specific
# (references ccdc_breaks' own column names) but shared so the extraction loop (which decides
# which points' raw time series to keep) and the standalone preview cell (which decides which
# cached time series to show) always agree on the same picks.
EXAMPLE_POINT_CRITERIA = {
    'random high-severity: crashes at ignition, recovers':
        lambda r: r['sample_type'] == 'random' and r['severity_name'] == 'high' and pd.notna(r['break_year']),
    'pre-fire outlier: real step-down before the fire':
        lambda r: pd.notna(r['break_year']) and r['break_year'] < 2020,
    'stratified Water, no break: no discontinuity anywhere':
        lambda r: r['sample_type'] == 'stratified_water' and pd.isna(r['break_year']),
    'stratified Developed: break confirmed near fire date':
        lambda r: r['sample_type'] == 'stratified_developed' and pd.notna(r['break_year']),
}


def pick_example_points(ccdc_breaks, criteria=EXAMPLE_POINT_CRITERIA):
    """First matching ccdc_breaks row per label in `criteria`.

    Returns {label: (index, row)}. `index` is ccdc_breaks' own DataFrame index, used as the
    lookup key into the pickled example time series saved alongside the PDF.
    """
    picked = {}
    for label, test in criteria.items():
        matches = ccdc_breaks[ccdc_breaks.apply(test, axis=1)]
        if not matches.empty:
            i = matches.index[0]
            picked[label] = (i, matches.loc[i])
    return picked


def hansen_loss_year(hansen_id=HANSEN_ID):
    """Hansen forest-loss year as a calendar-year image (2000 + lossyear).

    The raw `lossyear` band is 0 (no loss) or 1..N (loss in year 2000+N) at ~30 m. This
    returns 2000+lossyear, masked to loss pixels only (no-loss -> masked). Annual, and
    coarser than CCDC's sub-annual break dates -- a year-level cross-check, not a date one.
    """
    return ee.Image(hansen_id).select('lossyear').selfMask().add(2000).rename('loss_year')


def loss_year_table(region, mask=None, hansen_id=HANSEN_ID, scale=30):
    """True-ground-area (ha) of Hansen forest loss by year within region.

    Uses ee.Image.pixelArea() so the area is correct despite Hansen's EPSG:4326 grid (a plain
    pixel count would be latitude-distorted).

    Args:
        region: ee.Geometry to tabulate over.
        mask: Optional ee.Image mask (e.g. a burn-severity class) to further restrict to.
        hansen_id, scale: Hansen asset and reduction scale.

    Returns:
        DataFrame indexed by year, columns 'hectares' and 'fraction', sorted by year. For a
        single-year disturbance most area lands on that year, often with a one-year spill into
        the next (Hansen confirms late-season loss in the following year's imagery).
    """
    loss = hansen_loss_year(hansen_id)
    if mask is not None:
        loss = loss.updateMask(mask)
    groups = ee.List(
        ee.Image.pixelArea().divide(1e4).addBands(loss)
        .reduceRegion(ee.Reducer.sum().group(groupField=1, groupName='year'),
                      region, scale, maxPixels=1e10).get('groups')).getInfo()

    df = (pd.DataFrame([(int(g['year']), g['sum']) for g in groups], columns=['year', 'hectares'])
          .set_index('year').sort_index())
    df['fraction'] = df['hectares'] / df['hectares'].sum()
    return df
