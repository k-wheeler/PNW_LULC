import os

import ee
import numpy as np
import pandas as pd

EMBEDDING_COLLECTION_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
EMBEDDING_BANDS = [f'A{i:02d}' for i in range(64)]


def _sample_embeddings_for_year(df_year, year, batch_size=2000):
    """Sample the AlphaEarth annual embedding image at each (Lat, Lon) in df_year.
    Returns a DataFrame with Glance_ID, Year, and the 64 embedding bands (A00-A63).
    Points that fall on a masked/no-data pixel are silently dropped by sampleRegions,
    so the returned frame can have fewer rows than df_year."""
    image = (ee.ImageCollection(EMBEDDING_COLLECTION_ID)
             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
             .mosaic())

    rows = df_year[['Glance_ID', 'Lat', 'Lon']].to_records(index=False)
    results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        features = [
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {'Glance_ID': str(gid)})
            for gid, lat, lon in batch
        ]
        fc = ee.FeatureCollection(features)
        sampled = image.sampleRegions(collection=fc, scale=10, tileScale=16, geometries=False)
        for feat in sampled.getInfo()['features']:
            props = feat['properties']
            props['Year'] = year
            results.append(props)
        print(f'  {year}: sampled {min(start + batch_size, len(rows))}/{len(rows)}')

    return pd.DataFrame(results)


def get_embeddings(unique_year_points, cache_dir, download_embeddings=True):
    """Get AlphaEarth embeddings for every (Glance_ID, Year, Lat, Lon) row in
    unique_year_points, caching one CSV per year under cache_dir.

    If a year's cache file already exists it is loaded from disk instead of
    re-sampling from GEE. If download_embeddings is False, years without an
    existing cache file are skipped (not sampled) rather than downloaded.
    """
    os.makedirs(cache_dir, exist_ok=True)

    frames = []
    for year in sorted(unique_year_points['Year'].unique()):
        year = int(year)
        cache_path = os.path.join(cache_dir, f'embeddings_{year}.csv')

        if os.path.exists(cache_path):
            print(f'Loading cached embeddings for {year} from {cache_path}')
            frames.append(pd.read_csv(cache_path))
            continue

        if not download_embeddings:
            print(f'No cache for {year} and download_embeddings=False -- skipping')
            continue

        df_year = unique_year_points.loc[unique_year_points['Year'] == year]
        print(f'Sampling {len(df_year)} points for {year}...')
        embeddings_year = _sample_embeddings_for_year(df_year, year)
        embeddings_year.to_csv(cache_path, index=False)
        frames.append(embeddings_year)

    if not frames:
        return pd.DataFrame(columns=['Glance_ID', 'Year'] + EMBEDDING_BANDS)

    return pd.concat(frames, ignore_index=True)


def _sample_patch_arrays_for_year(df_year, year, radius=2, batch_size=500):
    """Sample the full k x k x 64 embedding window around each point, for a CNN.

    Uses neighborhoodToArray so the raw spatial arrangement is preserved (unlike
    an aggregated mean/std, which a CNN needs to learn from). k = 2*radius + 1,
    so radius=2 -> 5x5. Transfers k*k*64 numbers per point, so batch_size is
    smaller than the single-pixel sampler.

    Returns:
        Tuple of (patches, keys):
          patches: float32 array of shape (n, k, k, 64); rows whose window could
                   not be read fully are filled with NaN.
          keys:    DataFrame with Glance_ID and Year, one row per patch.
    """
    image = (ee.ImageCollection(EMBEDDING_COLLECTION_ID)
             .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
             .mosaic())

    kernel = ee.Kernel.square(radius=radius, units='pixels')
    array_image = image.neighborhoodToArray(kernel)  # each band -> a k x k array per pixel
    k = 2 * radius + 1

    rows = df_year[['Glance_ID', 'Lat', 'Lon']].to_records(index=False)
    patches, gids = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        features = [
            ee.Feature(ee.Geometry.Point([float(lon), float(lat)]), {'Glance_ID': str(gid)})
            for gid, lat, lon in batch
        ]
        fc = ee.FeatureCollection(features)
        sampled = array_image.sampleRegions(collection=fc, scale=10, tileScale=16, geometries=False)
        for feat in sampled.getInfo()['features']:
            props = feat['properties']
            try:
                # Stack the 64 per-band k x k arrays into a (k, k, 64) window.
                window = np.stack([np.asarray(props[b], dtype=np.float32) for b in EMBEDDING_BANDS], axis=-1)
                if window.shape != (k, k, 64):
                    window = np.full((k, k, 64), np.nan, dtype=np.float32)
            except (KeyError, ValueError):
                window = np.full((k, k, 64), np.nan, dtype=np.float32)
            patches.append(window)
            gids.append(props['Glance_ID'])
        print(f'  {year}: sampled {min(start + batch_size, len(rows))}/{len(rows)}')

    patch_array = np.stack(patches, axis=0) if patches else np.empty((0, k, k, 64), dtype=np.float32)
    keys = pd.DataFrame({'Glance_ID': gids, 'Year': year})
    return patch_array, keys


def get_patch_arrays(unique_year_points, cache_dir, radius=2, download_embeddings=True):
    """Patch (window) version of get_embeddings for the CNN: per (Glance_ID, Year,
    Lat, Lon) row, get the raw k x k x 64 embedding window, caching one .npz per
    year under cache_dir (filenames include the radius so windows don't clash).

    Returns:
        Tuple of (patches, keys):
          patches: float32 array (n, k, k, 64) concatenated across years.
          keys:    DataFrame with Glance_ID and Year, aligned row-for-row.
    """
    os.makedirs(cache_dir, exist_ok=True)
    k = 2 * radius + 1

    patch_frames, key_frames = [], []
    for year in sorted(unique_year_points['Year'].unique()):
        year = int(year)
        cache_path = os.path.join(cache_dir, f'patches_r{radius}_{year}.npz')

        if os.path.exists(cache_path):
            print(f'Loading cached patches for {year} from {cache_path}')
            data = np.load(cache_path, allow_pickle=True)
            patch_frames.append(data['patches'])
            key_frames.append(pd.DataFrame({'Glance_ID': data['Glance_ID'], 'Year': data['Year']}))
            continue

        if not download_embeddings:
            print(f'No patch cache for {year} and download_embeddings=False -- skipping')
            continue

        df_year = unique_year_points.loc[unique_year_points['Year'] == year]
        print(f'Sampling {len(df_year)} patches for {year} (radius={radius})...')
        patches, keys = _sample_patch_arrays_for_year(df_year, year, radius=radius)
        np.savez_compressed(cache_path, patches=patches,
                            Glance_ID=keys['Glance_ID'].values, Year=keys['Year'].values)
        patch_frames.append(patches)
        key_frames.append(keys)

    if not patch_frames:
        return np.empty((0, k, k, 64), dtype=np.float32), pd.DataFrame(columns=['Glance_ID', 'Year'])

    return np.concatenate(patch_frames, axis=0), pd.concat(key_frames, ignore_index=True)


def align_patch_arrays(patches, keys, target_key_df):
    """Reorder patch windows to match target_key_df's (Glance_ID, Year) rows.

    Returns a float32 array with one window per target row (NaN window where a
    patch is missing), so it lines up positionally with the single-pixel table
    and can be indexed by the same train/test split.
    """
    position = {(str(g), int(y)): i for i, (g, y) in enumerate(zip(keys['Glance_ID'], keys['Year']))}
    window_shape = patches.shape[1:]
    out = np.full((len(target_key_df),) + window_shape, np.nan, dtype=np.float32)
    for row, (g, y) in enumerate(zip(target_key_df['Glance_ID'], target_key_df['Year'])):
        idx = position.get((str(g), int(y)))
        if idx is not None:
            out[row] = patches[idx]
    return out


def neighborhood_stats(patches):
    """Per-band neighborhood mean and std over each k x k window.

    Turns patch windows (n, k, k, 64) into tabular spatial-context features for a
    tree model: the mean captures local average, the std captures texture /
    heterogeneity. Masked pixels within a window are ignored (nan-aware); a
    fully-missing window yields NaN (which XGBoost handles natively).

    Returns:
        DataFrame (n, 128) with columns A00_mean..A63_mean, A00_std..A63_std.
    """
    n, _, _, channels = patches.shape
    flat = patches.reshape(n, -1, channels)  # (n, k*k, channels)
    with np.errstate(invalid='ignore'):  # all-NaN window -> NaN, no warning spam
        means = np.nanmean(flat, axis=1)
        stds = np.nanstd(flat, axis=1)
    columns = [f'{b}_mean' for b in EMBEDDING_BANDS] + [f'{b}_std' for b in EMBEDDING_BANDS]
    return pd.DataFrame(np.hstack([means, stds]), columns=columns)
