"""Activity-data tabulation for the carbon accounting (see the plan's "Aboveground carbon"
section and Pilot_LaborDayFires.ipynb).

Carbon is accounted as Activity Data x Emission Factor: area of forest lost, per reporting
domain (county), times carbon density (Mg C/ha). This module handles the *area* side --
attributing a boolean change raster to polygons (counties) by rasterizing them onto the same
grid the change was computed on, so the split reconciles exactly with the total.

Deliberately map-based counting (not the sample-based Olofsson estimator): simpler, but the
areas are biased and carry no confidence interval. See the plan's change-detection warning.
"""

import numpy as np
import pandas as pd
from rasterio.features import rasterize
from Constants import IPCC_CARBON_FRACTION

# FIA reports aboveground biomass in short tons/acre; convert to Mg/ha:
#   1 short ton = 0.907185 Mg,  1 acre = 0.404686 ha  ->  x 0.907185 / 0.404686
def convert_tons_acre_to_mg_ha(biomass):
    return biomass * 0.907185 / 0.404686  # ~2.2417

def area_by_polygons(mask, transform, crs, polygons_gdf, label_col='NAME', pixel_ha=None):
    """Area (ha) of a boolean raster `mask` attributed to each polygon.

    Rasterizes polygons_gdf onto the mask's exact grid, assigning each pixel to at most one
    polygon, then sums the True pixels of `mask` within each. Because it uses the mask's own
    transform, the per-polygon areas reconcile with the total (verify with `outside_ha`).

    Args:
        mask: 2-D boolean array (e.g. Trees->non-Trees change) in an equal-area CRS.
        transform: The mask's affine transform (rasterio).
        crs: The mask's CRS (e.g. 'EPSG:5070'); polygons_gdf is reprojected to it.
        polygons_gdf: GeoDataFrame of reporting polygons (e.g. Geo_Utils.get_counties()).
        label_col: Column to index the result by. Use a UNIQUE key -- 'GEOID' for OR+WA
            counties, since 'NAME' collides across the two states.
        pixel_ha: Hectares per pixel. If None, derived from the transform (assumes an
            equal-area CRS in metres, so |a*e| m^2 / 1e4).

    Returns:
        (DataFrame indexed by label_col with 'hectares' and 'fraction', outside_ha) where
        outside_ha is masked area falling in no polygon -- should be ~0 if the polygons cover
        the mask, and is the reconciliation check.
    """
    if pixel_ha is None:
        pixel_ha = abs(transform.a * transform.e) / 1e4

    polys = polygons_gdf.to_crs(crs)
    labels = list(polys[label_col])
    # Codes 1..n; 0 is reserved for "outside every polygon". Adjacent polygons don't overlap,
    # so each pixel gets exactly one code (rasterize takes the last on any tie).
    codes = {lab: i + 1 for i, lab in enumerate(labels)}
    burned = rasterize(
        ((geom, codes[lab]) for lab, geom in zip(labels, polys.geometry)),
        out_shape=mask.shape, transform=transform, fill=0, dtype='int32')

    rows = {}
    for lab, code in codes.items():
        rows[lab] = float((mask & (burned == code)).sum()) * pixel_ha
    df = pd.DataFrame({'hectares': rows})
    df.index.name = label_col
    total = float(mask.sum()) * pixel_ha
    df['fraction'] = df['hectares'] / total if total else np.nan
    df = df.sort_values('hectares', ascending=False)

    outside_ha = float((mask & (burned == 0)).sum()) * pixel_ha
    return df, outside_ha


def carbon_from_area(area_df, biomass_ef, carbon_fraction=IPCC_CARBON_FRACTION,
                     hectares_col='hectares'):
    """Aboveground carbon lost = area (ha) x biomass EF (Mg/ha) x carbon fraction, per domain.

    The Activity-Data x Emission-Factor multiply. Area is the map-counted change area
    (`area_by_polygons`); the emission factor is FIA aboveground *biomass* density per domain.

    **Uncertainty is EF-only.** FIA publishes a sampling error on biomass, so the EF term has a
    real SE; the area term is map-counted and has *no* CI (and is biased high -- see the plan's
    change-detection warning). So the reported +/- reflects EF uncertainty alone and understates
    total uncertainty. Do not present it as a full confidence interval.

    Args:
        area_df: DataFrame indexed by domain (e.g. county GEOID/NAME) with a hectares column,
            from area_by_polygons().
        biomass_ef: dict domain -> (agb_mg_ha, agb_se_mg_ha): aboveground biomass density and
            its sampling standard error, both Mg/ha. Domains must match area_df's index.
            (Use TONS_PER_ACRE_TO_MG_PER_HA to convert FIA's tons/acre.)
        carbon_fraction: biomass -> carbon (default IPCC 0.47).
        hectares_col: name of the area column in area_df.

    Returns:
        DataFrame indexed by domain (+ a 'TOTAL' row) with columns:
          hectares, agb_mg_ha, agb_se_mg_ha, carbon_mg (total Mg C lost),
          carbon_se_mg (EF-only SE). Domains in area_df but missing from biomass_ef get NaN
          EF and are flagged (their carbon is NaN, so the TOTAL excludes them -- fill them in).
    """
    rows = {}
    for domain, r in area_df.iterrows():
        ha = float(r[hectares_col])
        agb, se = biomass_ef.get(domain, (np.nan, np.nan))
        carbon = ha * agb * carbon_fraction               # Mg C (area treated as exact)
        carbon_se = ha * se * carbon_fraction              # EF-only SE
        rows[domain] = {'hectares': ha, 'agb_mg_ha': agb, 'agb_se_mg_ha': se,
                        'carbon_mg': carbon, 'carbon_se_mg': carbon_se}
    df = pd.DataFrame(rows).T
    df.index.name = area_df.index.name

    # Grand total: carbon sums; independent county SEs add in quadrature.
    total = {
        'hectares': df['hectares'].sum(),
        'agb_mg_ha': np.nan, 'agb_se_mg_ha': np.nan,
        'carbon_mg': df['carbon_mg'].sum(),
        'carbon_se_mg': np.sqrt((df['carbon_se_mg'] ** 2).sum()),
    }
    df.loc['TOTAL'] = total
    return df
