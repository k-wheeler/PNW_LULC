"""Shallow MLP head for classifying AlphaEarth embeddings (PyTorch).

Kept in its own module so Random_Forest.py stays torch-free. The public entry
point is fit_mlp(), which returns a picklable, sklearn-style wrapper whose
predict() returns the original GLanCE class labels -- interchangeable with the
RandomForest / XGBoost models so it works with evaluate_model and compare_models.
"""

import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _MLPNet(nn.Module):
    """64 -> 128 -> 64 -> n_classes with BatchNorm + ReLU + Dropout per hidden layer."""

    def __init__(self, input_dim, hidden_dims, n_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _MLPClassifier:
    """Wraps a trained _MLPNet + its LabelEncoder + StandardScaler so predict()
    returns original labels. Inference runs on CPU (the net is stored on CPU),
    which keeps joblib.dump/load device-agnostic and is plenty fast for scoring.
    """

    def __init__(self, model, label_encoder, scaler, classes_):
        self.model = model
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.classes_ = classes_

    def _logits(self, X):
        X = self.scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.from_numpy(X)).numpy()

    def predict(self, X):
        idx = self._logits(X).argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    def predict_proba(self, X):
        logits = self._logits(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def fit_mlp(x_train, y_train, groups=None, val_fraction=0.2, hidden_dims=(128, 64),
            dropout=0.3, epochs=50, batch_size=256, lr=1e-3, weight_decay=0.0,
            balance_classes=True, random_state=1234, device=None, verbose=False):
    """Train a shallow MLP on the embedding features.

    Args:
        x_train, y_train: Training predictors and labels (original GLanCE IDs).
        groups: Optional group labels (Glance_ID) aligned row-for-row with
            x_train. If given, a grouped hold-out validation split (val_fraction)
            is used to record train/val loss per epoch on the returned model's
            .history_ (for overfitting diagnostics); the model is then trained on
            the remaining rows. If None, trains on all rows and .history_['val']
            is None.
        val_fraction: Fraction of groups held out for the validation curve.
        hidden_dims: Sizes of the hidden layers (default (128, 64)).
        dropout: Dropout probability applied after each hidden layer.
        epochs, batch_size, lr, weight_decay: Training hyperparameters.
        balance_classes: If True (default), weight the loss by inverse class
            frequency to mirror the tree models' class_weight='balanced'.
        random_state: Seed for reproducibility.
        device: Torch device; defaults to MPS if available else CPU.
        verbose: If True, print the average loss every 10 epochs.

    Returns:
        Tuple of (fitted classifier whose predict() returns original labels,
        training time in seconds).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    x = np.asarray(x_train, dtype=np.float32)
    # Encode labels on the full set so all classes are represented.
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    n_classes = len(label_encoder.classes_)

    # Optional grouped hold-out split for the loss curve.
    if groups is not None and val_fraction and val_fraction > 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
        fit_idx, val_idx = next(splitter.split(x, y_encoded, groups=np.asarray(groups)))
    else:
        fit_idx, val_idx = np.arange(len(x)), None

    # Fit the scaler on the training portion only (no leakage into val loss).
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x[fit_idx]).astype(np.float32)
    train_ds = TensorDataset(torch.from_numpy(x_fit),
                             torch.from_numpy(y_encoded[fit_idx].astype(np.int64)))
    # drop_last avoids a size-1 final batch, which BatchNorm1d cannot handle.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    if val_idx is not None:
        x_val = scaler.transform(x[val_idx]).astype(np.float32)
        val_ds = TensorDataset(torch.from_numpy(x_val),
                               torch.from_numpy(y_encoded[val_idx].astype(np.int64)))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        val_ds = val_loader = None

    model = _MLPNet(x.shape[1], list(hidden_dims), n_classes, dropout).to(device)

    # Class weights from the full training labels (all classes present).
    if balance_classes:
        weights = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_encoded)
        loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        loss_weight = None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {'train': [], 'val': ([] if val_loader is not None else None),
               'xlabel': 'Epoch', 'ylabel': 'Loss'}

    start_time = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        history['train'].append(running / len(train_ds))

        if val_loader is not None:
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_running += criterion(model(xb), yb).item() * len(xb)
            history['val'].append(val_running / len(val_ds))

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"epoch {epoch + 1}/{epochs}  train {history['train'][-1]:.4f}"
            if val_loader is not None:
                msg += f"  val {history['val'][-1]:.4f}"
            print(msg)
    train_time = time.perf_counter() - start_time

    # Move to CPU so the wrapper pickles/loads regardless of MPS availability.
    model.to('cpu').eval()
    clf = _MLPClassifier(model, label_encoder, scaler, label_encoder.classes_)
    clf.history_ = history
    return clf, train_time
