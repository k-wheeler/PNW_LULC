"""Shared persistence + comparison-loader helpers for the model variants in Main.ipynb.

Every variant (Random Forest, XGBoost, their tuned/spatial versions, MLP, CNN) is saved
under a `{base}_model{ext}.joblib` / `{base}_training_time{ext}.txt` naming scheme (plus
importances/confusion/losscurve/valcurve PNGs). These helpers factor out that scheme so
both the Level-1 (`Model_Outputs/`) and Level-2 (`Model_Outputs/Class2/`) sections of the
notebook can fit-or-load and reload-for-comparison without duplicating the boilerplate.
"""

import os

import joblib

# display name -> (filename base, filename ext), matching the {base}_model{ext}.joblib /
# {base}_training_time{ext}.txt scheme used by every fit-or-load cell.
MODEL_VARIANTS = [
    ('Random Forest',      'random_forest',       ''),
    ('XGBoost',            'XGBoost',              ''),
    ('RF (tuned)',         'random_forest_tuned',  '_groupedCV'),
    ('XGBoost (tuned)',    'XGBoost_tuned',        '_groupedCV'),
    ('MLP',                'MLP',                  ''),
    ('MLP (tuned)',        'MLP',                  '_tuned'),
    ('CNN',                'CNN',                  ''),
    ('CNN (tuned)',        'CNN',                  '_tuned'),
    ('XGBoost (spatial)',  'XGBoost_nbhd_tuned',   '_spatial'),
]

# Variants whose predict() input differs from the shared single-pixel x_test.
_PATCH_MODELS = {'CNN', 'CNN (tuned)'}
_NBHD_MODELS = {'XGBoost (spatial)'}


def variant_paths(out_dir, base, ext=""):
    """Return the standard {base}_{suffix}{ext} output paths for one model variant."""
    return {
        'model': os.path.join(out_dir, f'{base}_model{ext}.joblib'),
        'time': os.path.join(out_dir, f'{base}_training_time{ext}.txt'),
        'importances': os.path.join(out_dir, f'{base}_importances{ext}.png'),
        'confusion': os.path.join(out_dir, f'{base}_confusion{ext}.png'),
        'losscurve': os.path.join(out_dir, f'{base}_losscurve{ext}.png'),
        'valcurve': os.path.join(out_dir, f'{base}_valcurve{ext}.png'),
    }


def fit_or_load(model_path, time_path, fit_fn, refit_models):
    """Fit-or-load a model, saving it (and its training time) on a fresh fit.

    Args:
        model_path, time_path: Paths from variant_paths()['model'] / ['time'].
        fit_fn: Zero-arg callable returning (model, train_time) or
            (model, train_time, best_params) -- the fit_*/tune_* functions in
            Tree_Ensemble.py/MLP.py/CNN.py wrapped in a lambda by the caller.
        refit_models: If True, or if model_path doesn't exist yet, call fit_fn()
            and save the result; otherwise load the existing model + time.

    Returns:
        Tuple of (fitted model, training time in seconds).
    """
    if refit_models or not os.path.exists(model_path):
        result = fit_fn()
        mdl, training_time_sec = result[0], result[1]
        best_params = result[2] if len(result) > 2 else None

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(mdl, model_path)
        print(f'Model saved to {model_path}')
        with open(time_path, 'w') as f:
            f.write(f'{training_time_sec:.2f} seconds\n')
            if best_params is not None:
                f.write(f'best_params: {best_params}\n')
    else:
        mdl = joblib.load(model_path)
        with open(time_path) as f:
            training_time_sec = float(f.read().split()[0])

    return mdl, training_time_sec


def load_all_models(out_dir, patch_test, nbhd_test):
    """Reload every saved variant from out_dir for compare_models().

    Args:
        out_dir: Directory the variants were saved to (MODEL_DIR or its Class2/ subfolder).
        patch_test: Patch-array test input for the CNN variants (X_test_patch, or its
            WA/OR-masked subset).
        nbhd_test: Neighborhood-feature test input for XGBoost (spatial) (x_test_nbhd, or
            its WA/OR-masked subset).

    Returns:
        Dict mapping display name -> (model, train_time) or, for the patch/nbhd variants,
        (model, train_time, x_test_override) -- ready to pass to compare_models().
    """
    models = {}
    for name, base, ext in MODEL_VARIANTS:
        paths = variant_paths(out_dir, base, ext)
        mdl = joblib.load(paths['model'])
        with open(paths['time']) as f:
            training_time_sec = float(f.read().split()[0])

        if name in _PATCH_MODELS:
            models[name] = (mdl, training_time_sec, patch_test)
        elif name in _NBHD_MODELS:
            models[name] = (mdl, training_time_sec, nbhd_test)
        else:
            models[name] = (mdl, training_time_sec)
    return models
