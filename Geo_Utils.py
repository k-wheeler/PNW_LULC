"""Washington/Oregon geometry helpers.

Used by the Class Distribution and Model Comparison (WA/OR subset) cells for both the
Level-1 and Level-2 sections of Main.ipynb, and by Map_Export.py to build the
wall-to-wall export AOI.
"""

import ee
import geopandas as gpd

STATES_COLLECTION_ID = 'TIGER/2018/States'
STATE_NAMES = ['Washington', 'Oregon']


def _wa_or_fc():
    """The two states as an Earth Engine FeatureCollection."""
    return (ee.FeatureCollection(STATES_COLLECTION_ID)
            .filter(ee.Filter.inList('NAME', STATE_NAMES)))


def get_wa_or_states():
    """Washington + Oregon polygons from TIGER 2018 state boundaries.

    Pulls the geometry client-side (one getInfo) for point-in-polygon work; see
    get_wa_or_geometry() for the server-side counterpart.
    """
    return gpd.GeoDataFrame.from_features(
        _wa_or_fc().getInfo()['features'], crs='EPSG:4326')[['NAME', 'geometry']]


def get_wa_or_geometry(max_error=100):
    """Washington + Oregon dissolved into a single server-side ee.Geometry.

    The Earth Engine counterpart to get_wa_or_states(): it stays server-side, so it can be
    used directly as an Export region / intersection geometry without pulling the (large)
    state polygons down to the client.

    Args:
        max_error: Maximum reprojection error in metres allowed by dissolve().
    """
    return _wa_or_fc().geometry().dissolve(maxError=max_error)


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
