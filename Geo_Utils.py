"""Washington/Oregon geometry helpers shared by the Class Distribution and Model
Comparison (WA/OR subset) cells for both the Level-1 and Level-2 sections of Main.ipynb.
"""

import ee
import geopandas as gpd


def get_wa_or_states():
    """Washington + Oregon polygons from TIGER 2018 state boundaries."""
    wa_or_fc = (ee.FeatureCollection('TIGER/2018/States')
                .filter(ee.Filter.inList('NAME', ['Washington', 'Oregon'])))
    return gpd.GeoDataFrame.from_features(
        wa_or_fc.getInfo()['features'], crs='EPSG:4326')[['NAME', 'geometry']]


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
