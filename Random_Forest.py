import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold, validation_curve,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

# Default hyperparameter search spaces for the grouped-CV tuners below.
RF_PARAM_DISTRIBUTIONS = {
    'n_estimators': [200, 400, 600, 800],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.5],
}

XGB_PARAM_DISTRIBUTIONS = {
    'n_estimators': [200, 400, 600],
    'max_depth': [3, 4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_lambda': [0.5, 1, 2, 5],
}

def fit_random_forest(x_train, y_train, **kwargs):
    """Fit a random forest classifier on the training data.

    Args:
        x_train: DataFrame of predictors.
        y_train: Series of response labels.
        **kwargs: Additional keyword arguments passed to RandomForestClassifier
            (e.g. n_estimators, max_depth), overriding the defaults below.

    Returns:
        Tuple of (fitted RandomForestClassifier, training time in seconds).
    """
    params = {'class_weight': 'balanced', 'random_state': 1234}
    params.update(kwargs)

    model = RandomForestClassifier(**params)

    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    train_time = time.perf_counter() - start_time

    return model, train_time


class _LabelDecodingClassifier:
    """Wraps a classifier trained on label-encoded targets so that predict()
    returns the original (un-encoded) labels, keeping it interchangeable with
    the RandomForest model (whose predict() already returns original labels).

    XGBoost >= 1.6 requires class labels to be a contiguous 0..k-1 range, but
    the GLanCE class IDs are 1-indexed and non-contiguous (Level 2 has no class
    2 in North America), so they are LabelEncoded before fitting and decoded on
    the way back out here.
    """

    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        # Columns are ordered to match self.classes_ (sorted original labels).
        return self.model.predict_proba(X)

    def __getattr__(self, name):
        # Delegate anything else (e.g. feature_importances_) to the wrapped
        # model, but never intercept dunders so pickling/joblib.dump still work.
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self.model, name)


def fit_xgboost(x_train, y_train, groups=None, val_fraction=0.2, balance_classes=True, **kwargs):
    """Fit an XGBoost classifier on the training data.

    Kept separate from fit_random_forest so the plain RandomForest path still
    works even if xgboost is not installed (xgboost is imported lazily here).

    Args:
        x_train: DataFrame of predictors.
        y_train: Series of response labels (original GLanCE class IDs).
        groups: Optional group labels (Glance_ID) aligned row-for-row with
            x_train. If given, a grouped hold-out validation split (val_fraction)
            records train/val mlogloss per boosting round on the returned model's
            .history_ (for overfitting / early-stopping diagnostics); the model
            is then trained on the remaining rows. If None, trains on all of
            x_train and .history_ is None.
        val_fraction: Fraction of groups held out for the validation curve.
        balance_classes: If True (default), pass per-sample 'balanced' weights
            to mirror RandomForest's class_weight='balanced' (XGBoost has no
            class_weight for multiclass).
        **kwargs: Additional keyword arguments passed to XGBClassifier
            (e.g. n_estimators, max_depth, learning_rate), overriding defaults.

    Returns:
        Tuple of (fitted classifier whose predict() returns original labels,
        training time in seconds).
    """
    from xgboost import XGBClassifier

    params = {'random_state': 1234, 'eval_metric': 'mlogloss'}
    params.update(kwargs)

    # Encode labels to a contiguous 0..k-1 range as XGBoost requires.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # Optional grouped hold-out split so we can record a train/val loss curve.
    if groups is not None and val_fraction and val_fraction > 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction,
                                     random_state=params['random_state'])
        fit_idx, val_idx = next(splitter.split(x_train, y_encoded, groups=np.asarray(groups)))
        x_fit, y_fit = x_train.iloc[fit_idx], y_encoded[fit_idx]
        eval_set = [(x_fit, y_fit), (x_train.iloc[val_idx], y_encoded[val_idx])]
    else:
        x_fit, y_fit = x_train, y_encoded
        eval_set = None

    sample_weight = compute_sample_weight('balanced', y_fit) if balance_classes else None

    model = XGBClassifier(**params)

    start_time = time.perf_counter()
    model.fit(x_fit, y_fit, sample_weight=sample_weight, eval_set=eval_set, verbose=False)
    train_time = time.perf_counter() - start_time

    wrapper = _LabelDecodingClassifier(model, label_encoder)
    if eval_set is not None:
        results = model.evals_result_  # validation_0 = train split, validation_1 = val split
        wrapper.history_ = {
            'train': results['validation_0']['mlogloss'],
            'val': results['validation_1']['mlogloss'],
            'xlabel': 'Boosting round', 'ylabel': 'mlogloss',
        }
    else:
        wrapper.history_ = None
    return wrapper, train_time


def tune_random_forest(x_train, y_train, groups, param_distributions=None, n_iter=20,
                       n_splits=4, scoring='f1_macro', random_state=1234, n_jobs=-1):
    """Grouped-CV randomized hyperparameter search for a Random Forest.

    Uses StratifiedGroupKFold so all rows sharing a Glance_ID stay in the same
    fold (no leakage) while class balance is preserved across folds, and selects
    on macro-F1 by default so the imbalanced rare classes are not ignored.

    Args:
        x_train, y_train: Training predictors and labels.
        groups: Group labels (Glance_ID) aligned row-for-row with x_train.
        param_distributions: Dict of param -> list to sample from
            (defaults to RF_PARAM_DISTRIBUTIONS).
        n_iter: Number of random parameter combinations to try.
        n_splits: Number of grouped CV folds.
        scoring: Model-selection metric (default 'f1_macro').
        random_state: Seed for reproducibility.
        n_jobs: Parallel jobs for the search.

    Returns:
        Tuple of (best fitted RandomForestClassifier, total search time in
        seconds, best_params dict).
    """
    if param_distributions is None:
        param_distributions = RF_PARAM_DISTRIBUTIONS

    base = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base, param_distributions, n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=random_state, n_jobs=n_jobs, refit=True,
    )

    start_time = time.perf_counter()
    search.fit(x_train, y_train, groups=groups)
    search_time = time.perf_counter() - start_time

    return search.best_estimator_, search_time, search.best_params_


def tune_xgboost(x_train, y_train, groups, param_distributions=None, n_iter=20,
                 n_splits=4, scoring='f1_macro', random_state=1234, n_jobs=-1):
    """Grouped-CV randomized hyperparameter search for XGBoost.

    The search runs on label-encoded targets (XGBoost requires a contiguous
    0..k-1 range) with StratifiedGroupKFold and macro-F1 selection. The winning
    params are then refit on the full training set via fit_xgboost, so the final
    model is class-balanced, label-decoding, and picklable exactly like the
    plain XGBoost model.

    Args:
        x_train, y_train: Training predictors and labels (original GLanCE IDs).
        groups: Group labels (Glance_ID) aligned row-for-row with x_train.
        param_distributions: Dict of param -> list to sample from
            (defaults to XGB_PARAM_DISTRIBUTIONS).
        n_iter, n_splits, scoring, random_state, n_jobs: As in tune_random_forest.

    Returns:
        Tuple of (best fitted classifier whose predict() returns original labels,
        total time in seconds, best_params dict).
    """
    from xgboost import XGBClassifier

    if param_distributions is None:
        param_distributions = XGB_PARAM_DISTRIBUTIONS

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # n_jobs=1 on the estimator so the search parallelizes over folds instead of
    # each booster grabbing every core (avoids oversubscription).
    base = XGBClassifier(random_state=random_state, n_jobs=1)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        base, param_distributions, n_iter=n_iter, scoring=scoring, cv=cv,
        random_state=random_state, n_jobs=n_jobs, refit=False,
    )

    start_time = time.perf_counter()
    search.fit(x_train, y_encoded, groups=groups)

    # Refit the winning params with class balancing; pass groups so the tuned
    # model also gets a train/val boosting-round curve on its .history_.
    best_model, _ = fit_xgboost(x_train, y_train, groups=groups, **search.best_params_)
    total_time = time.perf_counter() - start_time

    return best_model, total_time, search.best_params_


def rf_validation_curve(x_train, y_train, groups, param_name='min_samples_leaf',
                        param_range=(1, 2, 4, 8, 16), n_splits=4, scoring='f1_macro',
                        random_state=1234, n_jobs=-1):
    """Grouped-CV validation curve for a Random Forest over one hyperparameter.

    A Random Forest has no epochs (more trees converge rather than overfit), so
    this is the overfitting stand-in for a loss curve: train vs cross-validation
    score as a pruning parameter varies. A wide train-minus-CV gap at small
    min_samples_leaf (or large max_depth) indicates overfitting.

    Args:
        x_train, y_train: Training predictors and labels.
        groups: Group labels (Glance_ID) aligned row-for-row with x_train.
        param_name: RandomForest parameter to vary (e.g. 'min_samples_leaf',
            'max_depth').
        param_range: Values of param_name to evaluate.
        n_splits, scoring, random_state, n_jobs: Grouped-CV settings.

    Returns:
        Tuple of (param_range, train_scores, val_scores) where each score array
        has shape (len(param_range), n_splits), as from validation_curve.
    """
    base = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    train_scores, val_scores = validation_curve(
        base, x_train, y_train, param_name=param_name, param_range=list(param_range),
        groups=groups, cv=cv, scoring=scoring, n_jobs=n_jobs,
    )
    return list(param_range), train_scores, val_scores


def split_data(feature_df, response_name, group_col='Glance_ID'):
    response = feature_df[response_name]
    groups = feature_df[group_col]
    predictor_cols = [c for c in feature_df.columns if c not in (response_name, group_col)]
    predictors = feature_df[predictor_cols]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1234)
    train_idx, test_idx = next(splitter.split(predictors, response, groups=groups))

    x_train, x_test = predictors.iloc[train_idx], predictors.iloc[test_idx]
    y_train, y_test = response.iloc[train_idx], response.iloc[test_idx]

    return x_train, x_test, y_train, y_test