"""MTBS fire helpers for the 2020 Labor Day fires pilot (see Pilot_LaborDayFires.ipynb).

MTBS (Monitoring Trends in Burn Severity) is the reference truth the pilot validates
against: a fire perimeter (known area), an ignition date (known change date), and a 30 m
burn-severity raster (which parts actually changed land-cover class).

Careful with area. The MTBS severity mosaic is served in EPSG:3338 (Alaska Albers), so at
Oregon's latitude a nominal 30 m pixel is NOT 900 m2 on the ground -- counting pixels x 900
overstates area by 1/cos(lat) ~= 1.39 here. Always use ee.Image.pixelArea() for true ground
area; severity_area_table() does.
"""

import datetime as dt

import ee
import pandas as pd

MTBS_BOUNDARIES_ID = 'USFS/GTAC/MTBS/burned_area_boundaries/v1'
MTBS_SEVERITY_ID = 'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1'

# MTBS Severity band classes.
SEVERITY_NAMES = {
    1: 'unburned/low',
    2: 'low',
    3: 'moderate',
    4: 'high',
    5: 'increased greenness',
    6: 'non-mapping',
}
# The severity classes that plausibly flip a Trees pixel to another land-cover class. Low /
# unburned-low leave forest standing (still Trees), so a change map should detect roughly the
# moderate+high area, not the whole perimeter -- the pilot's central falsifiable prediction.
STAND_CHANGING_SEVERITY = [3, 4]


def get_fire(incid_name, min_acres=100000):
    """The MTBS burned-area feature for one fire, by incident name.

    Args:
        incid_name: MTBS Incid_Name, e.g. 'HOLIDAY FARM' (upper-case in MTBS).
        min_acres: Guard against smaller same-named incidents; the Labor Day megafires are
            all >20k ac, so a high floor uniquely selects the intended event.

    Returns:
        An ee.Feature (the largest match above the floor).
    """
    fc = (ee.FeatureCollection(MTBS_BOUNDARIES_ID)
          .filter(ee.Filter.eq('Incid_Name', incid_name))
          .filter(ee.Filter.gt('BurnBndAc', min_acres))
          .sort('BurnBndAc', False))
    return ee.Feature(fc.first())


def fire_ignition_date(fire):
    """Ignition date of a fire feature as a Python date (MTBS Ig_Date is epoch-ms)."""
    ms = fire.get('Ig_Date').getInfo()
    return dt.datetime.utcfromtimestamp(ms / 1000).date()


def get_severity_image(year):
    """The MTBS burn-severity mosaic for one year (single 'Severity' band, 30 m)."""
    return (ee.ImageCollection(MTBS_SEVERITY_ID)
            .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
            .mosaic()
            .select('Severity'))


def severity_area_table(fire, year, scale=30):
    """True-ground-area (hectares) by burn-severity class within a fire perimeter.

    Uses ee.Image.pixelArea() so the result is correct despite the severity mosaic's
    EPSG:3338 projection (a plain pixel count would be inflated by ~1/cos(lat)).

    Returns:
        DataFrame indexed by severity name, columns 'hectares' and 'fraction', with a Total
        row. The moderate+high rows are the expected detectable-change area.
    """
    geom = fire.geometry()
    severity = get_severity_image(year)
    area_by_class = (ee.Image.pixelArea().divide(1e4)  # m2 -> ha
                     .addBands(severity)
                     .reduceRegion(
                         reducer=ee.Reducer.sum().group(groupField=1, groupName='severity'),
                         geometry=geom, scale=scale, maxPixels=1e10)
                     .get('groups'))
    groups = ee.List(area_by_class).getInfo()

    rows = {SEVERITY_NAMES.get(int(g['severity']), str(g['severity'])): g['sum'] for g in groups}
    df = pd.DataFrame({'hectares': rows})
    df['fraction'] = df['hectares'] / df['hectares'].sum()
    df.loc['Total'] = [df['hectares'].sum(), 1.0]
    return df


def stand_changing_mask(year):
    """Mask that is 1 where burn severity is stand-changing (moderate or high), else masked.

    Intersect this with the classifier's detected Trees->non-Trees change to score how well
    the map recovers the part of the fire that actually changed land-cover class.
    """
    severity = get_severity_image(year)
    return severity.remap(STAND_CHANGING_SEVERITY, [1] * len(STAND_CHANGING_SEVERITY), 0).selfMask()
