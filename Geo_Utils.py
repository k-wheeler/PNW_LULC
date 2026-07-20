"""Washington/Oregon geometry helpers.

Used by the Class Distribution and Model Comparison (WA/OR subset) cells for both the
Level-1 and Level-2 sections of Main.ipynb, and by Map_Export.py to build the
wall-to-wall export AOI.
"""

import ee
import geopandas as gpd

STATES_COLLECTION_ID = 'TIGER/2018/States'
STATE_NAMES = ['Washington', 'Oregon']

COUNTIES_COLLECTION_ID = 'TIGER/2018/Counties'
STATE_FIPS = ['41', '53']  # Oregon = 41, Washington = 53


def _wa_or_fc(state_names=STATE_NAMES):
    """One or both states as an Earth Engine FeatureCollection."""
    return (ee.FeatureCollection(STATES_COLLECTION_ID)
            .filter(ee.Filter.inList('NAME', list(state_names))))


def get_wa_or_states():
    """Washington + Oregon polygons from TIGER 2018 state boundaries.

    Pulls the geometry client-side (one getInfo) for point-in-polygon work; see
    get_wa_or_geometry() for the server-side counterpart.
    """
    return gpd.GeoDataFrame.from_features(
        _wa_or_fc().getInfo()['features'], crs='EPSG:4326')[['NAME', 'geometry']]


def get_counties(state_fps=STATE_FIPS, names=None):
    """OR/WA counties from TIGER 2018 as a GeoDataFrame (EPSG:4326).

    Used to attribute change/carbon to a reporting county (the pilot uses Lane + Linn; the
    full carbon run uses all OR+WA counties).

    Args:
        state_fps: State FIPS codes to include (default Oregon 41 + Washington 53).
        names: Optional list of county NAMEs to restrict to (e.g. ['Lane', 'Linn']).

    Returns:
        GeoDataFrame with NAME, STATEFP, GEOID, geometry. **Key on GEOID, not NAME** -- county
        names are not unique across OR+WA (both states have a Columbia and a Lincoln county);
        GEOID (state+county FIPS) is the unique identifier.
    """
    filt = ee.Filter.inList('STATEFP', list(state_fps))
    if names is not None:
        filt = ee.Filter.And(filt, ee.Filter.inList('NAME', list(names)))
    fc = ee.FeatureCollection(COUNTIES_COLLECTION_ID).filter(filt)
    gdf = gpd.GeoDataFrame.from_features(fc.getInfo()['features'], crs='EPSG:4326')
    return gdf[['NAME', 'STATEFP', 'GEOID', 'geometry']]


def get_wa_or_geometry(max_error=100, state_names=STATE_NAMES):
    """One or both states dissolved into a single server-side ee.Geometry.

    The Earth Engine counterpart to get_wa_or_states(): it stays server-side, so it can be
    used directly as an Export region / intersection geometry without pulling the (large)
    state polygons down to the client. Defaults to Washington + Oregon; pass
    state_names=['Oregon'] to restrict to a single state (e.g. for an Oregon-only wall-to-wall
    run -- see Map_Export.get_forest_aoi).

    Args:
        max_error: Maximum reprojection error in metres allowed by dissolve().
        state_names: State NAMEs to include (default both WA + OR).
    """
    return _wa_or_fc(state_names).geometry().dissolve(maxError=max_error)


def assign_wa_or_state(expanded_df, wa_or_states):
    """Point-in-polygon state ('Washington' / 'Oregon' / NaN) for each row of expanded_df.

    Args:
        expanded_df: DataFrame with 'Glance_ID', 'Lat', 'Lon' columns (e.g.
            expanded_Glance_Class1 or expanded_Glance_Class2).
        wa_or_states: GeoDataFrame from get_wa_or_states().

    Returns:
        Series of state names (or NaN outside WA/OR), aligned to expanded_df.index.
    """
    pts = gpd.GeoDataFrame(
        expanded_df[['Glance_ID']].copy(),
        geometry=gpd.points_from_xy(expanded_df['Lon'], expanded_df['Lat']),
        crs='EPSG:4326')
    joined = gpd.sjoin(pts, wa_or_states, predicate='within', how='left')
    joined = joined[~joined.index.duplicated(keep='first')]  # guard against border points matching both
    return joined['NAME'].reindex(expanded_df.index)
