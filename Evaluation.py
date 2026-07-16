"""Model-agnostic evaluation and plotting helpers.

Works for any fitted classifier that exposes `.predict()` and
`.feature_importances_` (Random Forest, the XGBoost wrapper, etc.), so the same
functions can be reused as more model types are added.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    ConfusionMatrixDisplay,
    f1_score,
    matthews_corrcoef,
)


def plot_feature_importances(model, feature_names, top_n=20, title='Feature Importances',
                             ax=None, save_path=None):
    """Bar plot of the most important features for a tree-based model.

    Args:
        model: Fitted model exposing `.feature_importances_` (RandomForest or the
            XGBoost wrapper).
        feature_names: List/Index of feature names, or the predictor DataFrame
            itself (its columns are used).
        top_n: Show only the top_n most important features.
        title: Plot title.
        ax: Optional existing matplotlib Axes to draw on.
        save_path: If given, save the figure to this path.

    Returns:
        The matplotlib Axes.
    """
    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns
    feature_names = np.asarray(feature_names)

    importances = np.asarray(model.feature_importances_)
    order = np.argsort(importances)[::-1][:top_n]
    # Reverse so the largest bar is at the top of a horizontal bar chart.
    order = order[::-1]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(order))))

    ax.barh(range(len(order)), importances[order], color='#4c72b0')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(feature_names[order])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.margins(y=0.01)

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight', dpi=150)

    return ax


def evaluate_model(model, x_test, y_test, class_map=None, normalize='true',
                   title='Confusion Matrix', ax=None, save_path=None, digits=3):
    """Evaluate a fitted classifier: print a per-class report + headline metrics
    and plot a confusion matrix.

    Because the GLanCE classes are imbalanced, this reports macro-F1 and balanced
    accuracy alongside overall accuracy so the rare classes are not hidden.

    Args:
        model: Fitted classifier whose `.predict()` returns original class labels.
        x_test, y_test: Held-out predictors and labels.
        class_map: Optional dict mapping class label -> readable name (e.g.
            class1_dict / class2_dict) used for the report and axis labels.
        normalize: Passed to the confusion matrix ('true' normalizes each row to
            recall; None shows raw counts).
        title: Confusion-matrix plot title.
        ax: Optional existing matplotlib Axes for the confusion matrix.
        save_path: If given, save the confusion-matrix figure to this path.
        digits: Decimal places in the classification report.

    Returns:
        Dict of headline metrics plus the raw predictions.
    """
    y_pred = model.predict(x_test)

    labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(y_pred)]))
    target_names = [str(class_map[l]) for l in labels] if class_map is not None else None

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, labels=labels, average='macro')
    # Chance-corrected agreement (kappa) and a balanced multiclass correlation
    # (MCC): both max at 1, and both stay honest under class imbalance.
    kappa = cohen_kappa_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=target_names, digits=digits, zero_division=0))
    print(f'Overall accuracy : {accuracy:.{digits}f}')
    print(f'Balanced accuracy: {balanced_accuracy:.{digits}f}')
    print(f'Macro F1         : {macro_f1:.{digits}f}')
    print(f"Cohen's kappa    : {kappa:.{digits}f}")
    print(f'MCC              : {mcc:.{digits}f}')

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, display_labels=target_names,
        normalize=normalize, cmap='Blues', xticks_rotation=45,
        values_format='.2f' if normalize else 'd', ax=ax, colorbar=False,
    )
    ax.set_title(title)
    disp.figure_.tight_layout()

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight', dpi=150)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'macro_f1': macro_f1,
        'cohen_kappa': kappa,
        'mcc': mcc,
        'y_pred': y_pred,
    }


def compare_models(models, x_test, y_test, class_map=None, save_dir=None,
                   prefix='model_comparison'):
    """Compare several fitted models on one held-out set: build summary tables and
    comparison figures side by side.

    Args:
        models: Dict mapping display name -> (fitted_model, training_time_seconds).
            Each model's predict() must return original class labels. A value may
            also be a 3-tuple (model, training_time, x_test_override) for a model
            whose input differs from the shared x_test (e.g. the CNN, which takes
            patch arrays); the override must correspond to the same rows as y_test.
        x_test, y_test: Shared held-out predictors and labels.
        class_map: Optional dict mapping class label -> readable name.
        save_dir: If given, save the summary CSV and figures here.
        prefix: Filename prefix for saved outputs.

    Returns:
        Tuple of (summary_df, per_class_f1_df).
          summary_df:      one row per model, headline metrics + training time.
          per_class_f1_df: per-class F1 (rows = classes, cols = models).
    """
    labels = np.unique(np.asarray(y_test))
    class_names = [str(class_map[l]) for l in labels] if class_map is not None else [str(l) for l in labels]

    summary = {}
    per_class = {}
    for name, spec in models.items():
        # spec is (model, train_time) using the shared x_test, or
        # (model, train_time, x_test_override) for models with a different input.
        model, train_time = spec[0], spec[1]
        x_eval = spec[2] if len(spec) > 2 else x_test
        y_pred = model.predict(x_eval)
        summary[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Balanced accuracy': balanced_accuracy_score(y_test, y_pred),
            'Macro F1': f1_score(y_test, y_pred, labels=labels, average='macro', zero_division=0),
            'Weighted F1': f1_score(y_test, y_pred, labels=labels, average='weighted', zero_division=0),
            "Cohen's kappa": cohen_kappa_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Training time (s)': train_time,
        }
        per_class[name] = f1_score(y_test, y_pred, labels=labels, average=None, zero_division=0)

    summary_df = pd.DataFrame(summary).T
    per_class_f1_df = pd.DataFrame(per_class, index=class_names)

    metric_cols = ['Accuracy', 'Balanced accuracy', 'Macro F1', 'Weighted F1']

    # ── Figure 1: headline metrics (grouped bars) + training time (own axis) ──
    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={'width_ratios': [3, 1]})
    summary_df[metric_cols].plot(kind='bar', ax=ax_m, colormap='viridis', width=0.8)
    ax_m.set_title('Accuracy metrics by model')
    ax_m.set_ylabel('Score')
    ax_m.set_ylim(0, 1)
    ax_m.set_xticklabels(summary_df.index, rotation=20, ha='right')
    ax_m.legend(loc='lower right', fontsize=8)
    ax_m.grid(axis='y', alpha=0.3)

    time_positions = range(len(summary_df.index))
    time_bars = ax_t.bar(time_positions, summary_df['Training time (s)'], color='#c44e52')
    ax_t.set_title('Training time')
    ax_t.set_ylabel('Seconds')
    ax_t.set_xticks(list(time_positions))
    ax_t.set_xticklabels(summary_df.index, rotation=20, ha='right')
    ax_t.bar_label(time_bars, fmt='%.1f', fontsize=8)
    ax_t.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    # ── Figure 2: per-class F1 heatmap (classes x models) ──
    fig_h, ax_h = plt.subplots(figsize=(1.6 * len(models) + 3, 0.5 * len(class_names) + 2))
    data = per_class_f1_df.values
    im = ax_h.imshow(data, cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
    ax_h.set_xticks(range(len(per_class_f1_df.columns)))
    ax_h.set_xticklabels(per_class_f1_df.columns, rotation=20, ha='right')
    ax_h.set_yticks(range(len(class_names)))
    ax_h.set_yticklabels(class_names)
    ax_h.set_title('Per-class F1 by model')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax_h.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                      color='white' if data[i, j] > 0.5 else 'black', fontsize=8)
    fig_h.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label='F1')
    fig_h.tight_layout()

    if save_dir is not None:
        summary_df.to_csv(f'{save_dir}/{prefix}_summary.csv')
        per_class_f1_df.to_csv(f'{save_dir}/{prefix}_per_class_f1.csv')
        fig.savefig(f'{save_dir}/{prefix}_metrics.png', bbox_inches='tight', dpi=150)
        fig_h.savefig(f'{save_dir}/{prefix}_per_class_f1.png', bbox_inches='tight', dpi=150)

    return summary_df, per_class_f1_df


def plot_training_curve(history, title='Training curve', ax=None, save_path=None):
    """Plot train vs validation loss over epochs / boosting rounds.

    Args:
        history: dict with 'train' (list), 'val' (list or None) and optional
            'xlabel'/'ylabel' -- as stored on a fitted model's .history_ by
            fit_mlp / fit_cnn / fit_xgboost. None if the model has no history
            (e.g. a Random Forest, or a model loaded from disk that predates this).
        title, ax, save_path: title / existing Axes / optional output path.

    Returns:
        The matplotlib Axes, or None if there was no history to plot.
    """
    if not history or not history.get('train'):
        print(f'{title}: no training history available '
              '(model has no epochs, or was fit without a validation split).')
        return None

    train = history['train']
    val = history.get('val')
    xlabel = history.get('xlabel', 'Iteration')
    ylabel = history.get('ylabel', 'Loss')

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(train) + 1), train, label='train', color='#4c72b0')
    if val:
        ax.plot(range(1, len(val) + 1), val, label='validation', color='#c44e52')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight', dpi=150)
    return ax


def plot_validation_curve(param_range, train_scores, val_scores, param_name,
                          scoring='score', title=None, ax=None, save_path=None):
    """Plot mean +/- std train and cross-validation score vs a hyperparameter.

    The Random-Forest stand-in for a loss curve: a wide train-minus-CV gap
    signals overfitting at that setting.

    Args:
        param_range: sequence of parameter values (may include None).
        train_scores, val_scores: arrays of shape (len(param_range), n_folds),
            as returned by sklearn.model_selection.validation_curve.
        param_name, scoring, title, ax, save_path: labels / output.

    Returns:
        The matplotlib Axes.
    """
    train_mean, train_std = np.mean(train_scores, axis=1), np.std(train_scores, axis=1)
    val_mean, val_std = np.mean(val_scores, axis=1), np.std(val_scores, axis=1)
    x = np.arange(len(param_range))

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, train_mean, 'o-', label='train', color='#4c72b0')
    ax.fill_between(x, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#4c72b0')
    ax.plot(x, val_mean, 'o-', label='cross-val', color='#c44e52')
    ax.fill_between(x, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#c44e52')
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in param_range])
    ax.set_xlabel(param_name)
    ax.set_ylabel(scoring)
    ax.set_title(title or f'Validation curve over {param_name}')
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path is not None:
        ax.figure.savefig(save_path, bbox_inches='tight', dpi=150)
    return ax
