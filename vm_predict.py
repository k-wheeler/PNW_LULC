"""Tile-by-tile land-cover inference, run INSIDE the container on a Compute Engine VM.

Not imported by any notebook. It streams the AlphaEarth embedding tiles that Earth Engine
exported to GCS, scores each with the tuned XGBoost model baked into the image, and writes
two small rasters back to GCS per tile:

  predictions/  uint8 GLanCE Level-1 class (1-7), NODATA where the embedding was masked
  confidence/   uint8 max class probability * 100 (0-100), same NODATA

Design points that matter:
  * One tile in memory at a time (download -> predict -> upload -> delete the local copy),
    so an e2-standard-4 is plenty regardless of how many tiles the export produced.
  * Band order is read from the model (`feature_names_in_`), never hardcoded -- the model
    is the single source of truth for which column is which, so a reordered export or a
    future re-train cannot silently corrupt predictions.
  * NODATA is 255 (outside the 1-7 class range and the 0-100 confidence range), and masked
    embedding pixels stay NODATA rather than being assigned a spurious class.

Usage (on the VM, inside the container):
    python vm_predict.py --bucket BUCKET --embeddings-prefix embeddings/2019 \
        --predictions-prefix predictions/2019 --confidence-prefix confidence/2019

The model path defaults to the copy baked into the image (/app/model/...); override with
--model-path for local testing.
"""

import argparse
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import rasterio
from google.cloud import storage

NODATA = 255  # outside both the 1-7 class range and the 0-100 confidence range
DEFAULT_MODEL_PATH = '/app/model/XGBoost_tuned_model_groupedCV.joblib'


def load_model(model_path):
    """Load the pickled model and return (model, feature_names).

    feature_names comes from the model itself (feature_names_in_), so the DataFrame we
    build for prediction is guaranteed to match the columns the model was trained on.
    """
    model = joblib.load(model_path)
    names = getattr(model, 'feature_names_in_', None)
    if names is None:
        raise ValueError('Model has no feature_names_in_; cannot guarantee band order.')
    return model, list(names)


def predict_tile(src_path, model, feature_names):
    """Score one embedding GeoTIFF.

    Returns (class_arr, confidence_arr, profile):
      class_arr:      uint8 (H, W), GLanCE class 1-7, NODATA elsewhere.
      confidence_arr: uint8 (H, W), max probability * 100, NODATA elsewhere.
      profile:        rasterio profile for writing the outputs (single band, uint8).
    """
    with rasterio.open(src_path) as src:
        bands = src.read()  # (n_bands, H, W), band i == feature_names[i] by export order
        profile = src.profile
        # A pixel is valid only where every band is unmasked (AlphaEarth coverage gap ->
        # the whole 64-vector is nodata, not a single band).
        if src.nodata is not None:
            valid = np.all(bands != src.nodata, axis=0)
        else:
            valid = ~np.any(np.isnan(bands), axis=0)

    n_bands, height, width = bands.shape
    if n_bands != len(feature_names):
        raise ValueError(f'Tile has {n_bands} bands but model expects {len(feature_names)}.')

    class_arr = np.full((height, width), NODATA, dtype=np.uint8)
    conf_arr = np.full((height, width), NODATA, dtype=np.uint8)

    valid_flat = valid.reshape(-1)
    if valid_flat.any():
        # (n_bands, H, W) -> (n_valid_pixels, n_bands), columns named to match the model.
        pixels = bands.reshape(n_bands, -1)[:, valid_flat].T
        X = pd.DataFrame(pixels, columns=feature_names)

        proba = model.predict_proba(X)                 # (n_valid, n_classes)
        pred = model.classes_[proba.argmax(axis=1)]    # original labels 1-7
        conf = (proba.max(axis=1) * 100).round().astype(np.uint8)

        class_flat = class_arr.reshape(-1)
        conf_flat = conf_arr.reshape(-1)
        class_flat[valid_flat] = pred.astype(np.uint8)
        conf_flat[valid_flat] = conf
        class_arr = class_flat.reshape(height, width)
        conf_arr = conf_flat.reshape(height, width)

    profile.update(dtype='uint8', count=1, nodata=NODATA,
                   compress='deflate', predictor=2)
    return class_arr, conf_arr, profile


def _write(path, array, profile):
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(array, 1)


def process_bucket(bucket_name, embeddings_prefix, predictions_prefix, confidence_prefix,
                   model_path=DEFAULT_MODEL_PATH, skip_existing=False):
    """Score every embedding tile under embeddings_prefix, one at a time.

    skip_existing makes the run resumable: a tile whose prediction is already in GCS is
    skipped. This lets a second VM finish a run that a max-run-duration backstop cut short
    (the backstop deletes the VM, but the tiles it already wrote stay in the bucket), without
    reprocessing everything -- the fresh VM only does the remainder.
    """
    model, feature_names = load_model(model_path)
    print(f'Loaded model: {len(feature_names)} features, classes {list(model.classes_)}')

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = [b for b in client.list_blobs(bucket_name, prefix=embeddings_prefix)
             if b.name.endswith('.tif')]
    print(f'{len(blobs)} embedding tiles under gs://{bucket_name}/{embeddings_prefix}'
          f'{" (resume: skipping tiles already predicted)" if skip_existing else ""}')

    for i, blob in enumerate(blobs, 1):
        base = os.path.basename(blob.name)
        if skip_existing and bucket.blob(f'{predictions_prefix}/{base}').exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            local_in = os.path.join(tmp, base)
            blob.download_to_filename(local_in)

            class_arr, conf_arr, profile = predict_tile(local_in, model, feature_names)

            local_pred = os.path.join(tmp, f'pred_{base}')
            local_conf = os.path.join(tmp, f'conf_{base}')
            _write(local_pred, class_arr, profile)
            _write(local_conf, conf_arr, profile)

            bucket.blob(f'{predictions_prefix}/{base}').upload_from_filename(local_pred)
            bucket.blob(f'{confidence_prefix}/{base}').upload_from_filename(local_conf)
        # tmp (and the ~268 MB input tile) is deleted here before the next iteration.
        print(f'[{i}/{len(blobs)}] {base}')

    print('Done.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bucket', required=True, help='GCS bucket name (no gs:// prefix)')
    p.add_argument('--embeddings-prefix', required=True,
                   help="e.g. 'embeddings/2019'")
    p.add_argument('--predictions-prefix', required=True,
                   help="e.g. 'predictions/2019'")
    p.add_argument('--confidence-prefix', required=True,
                   help="e.g. 'confidence/2019'")
    p.add_argument('--model-path', default=DEFAULT_MODEL_PATH,
                   help='Override the baked-in model path (for local testing).')
    p.add_argument('--skip-existing', action='store_true',
                   help='Resume: skip tiles whose prediction is already in GCS.')
    args = p.parse_args()
    process_bucket(args.bucket, args.embeddings_prefix, args.predictions_prefix,
                   args.confidence_prefix, model_path=args.model_path,
                   skip_existing=args.skip_existing)


if __name__ == '__main__':
    main()
