"""Small CNN on AlphaEarth embedding patches (PyTorch).

Unlike the tree/MLP heads, which see a single 64-vector per point, this model
takes the raw k x k x 64 window around each point (from Embedding_Utils.
get_patch_arrays) and learns spatial structure -- texture, edges, local context
-- with convolutions. It is a standalone variant, not a head bolted onto the
other models. The public entry point fit_cnn() returns a picklable, sklearn-style
wrapper whose predict() returns original GLanCE labels, so it plugs into
evaluate_model and compare_models like every other variant.
"""

import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _CNNNet(nn.Module):
    """Two conv blocks (channels-in -> 64 -> 128) + global pooling + an MLP head.

    Global average pooling makes it agnostic to the exact window size, and the
    final Linear layers are the classification head (a CNN is a conv feature
    extractor followed by an MLP-style head).
    """

    def __init__(self, in_channels, n_classes, dropout):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


class _CNNClassifier:
    """Wraps a trained _CNNNet + LabelEncoder + per-channel StandardScaler so
    predict() returns original labels. Inference runs on CPU (net stored on CPU)
    so joblib.dump/load is device-agnostic.
    """

    def __init__(self, model, label_encoder, scaler, classes_):
        self.model = model
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.classes_ = classes_

    def _to_input(self, X):
        # X: (n, k, k, 64) -> standardize per channel -> (n, 64, k, k) tensor.
        X = np.asarray(X, dtype=np.float32)
        n, kh, kw, c = X.shape
        flat = self.scaler.transform(X.reshape(-1, c))
        flat = np.nan_to_num(flat, nan=0.0).astype(np.float32)
        chan_first = flat.reshape(n, kh, kw, c).transpose(0, 3, 1, 2)
        return torch.from_numpy(np.ascontiguousarray(chan_first))

    def _logits(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(self._to_input(X)).numpy()

    def predict(self, X):
        idx = self._logits(X).argmax(axis=1)
        return self.label_encoder.inverse_transform(idx)

    def predict_proba(self, X):
        logits = self._logits(X)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


def fit_cnn(x_train, y_train, groups=None, val_fraction=0.2, dropout=0.3, epochs=40,
            batch_size=128, lr=1e-3, weight_decay=0.0, balance_classes=True,
            random_state=1234, device=None, verbose=False):
    """Train a small CNN on embedding patches.

    Args:
        x_train: Patch array of shape (n, k, k, 64) (from get_patch_arrays +
            align_patch_arrays). Rows that are all-NaN (patch missing) are
            dropped from training.
        y_train: Labels (original GLanCE IDs), aligned to x_train rows.
        dropout, epochs, batch_size, lr, weight_decay: Training hyperparameters.
        balance_classes: If True (default), weight the loss by inverse class
            frequency, mirroring the other variants' balancing.
        random_state, device, verbose: As in fit_mlp.

    Returns:
        Tuple of (fitted classifier whose predict() returns original labels,
        training time in seconds).
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if device is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    x = np.asarray(x_train, dtype=np.float32)
    y = np.asarray(y_train)
    g = np.asarray(groups) if groups is not None else None

    # Drop rows whose window could not be sampled (all NaN) -- nothing to learn.
    valid = ~np.isnan(x).all(axis=(1, 2, 3))
    x, y = x[valid], y[valid]
    if g is not None:
        g = g[valid]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    n_classes = len(label_encoder.classes_)

    # Optional grouped hold-out split for the loss curve.
    if g is not None and val_fraction and val_fraction > 0:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
        fit_idx, val_idx = next(splitter.split(x, y_encoded, groups=g))
    else:
        fit_idx, val_idx = np.arange(len(x)), None

    _, kh, kw, c = x.shape

    def to_chan_first(rows, fit_scaler):
        # standardize per channel then reshape (m, k, k, c) -> (m, c, k, k)
        flat = np.nan_to_num(x[rows], nan=0.0).reshape(-1, c)
        flat = scaler.fit_transform(flat) if fit_scaler else scaler.transform(flat)
        arr = flat.reshape(len(rows), kh, kw, c).transpose(0, 3, 1, 2).astype(np.float32)
        return np.ascontiguousarray(arr)

    scaler = StandardScaler()
    x_fit = to_chan_first(fit_idx, fit_scaler=True)
    train_ds = TensorDataset(torch.from_numpy(x_fit),
                             torch.from_numpy(y_encoded[fit_idx].astype(np.int64)))
    # drop_last avoids a size-1 final batch, which BatchNorm cannot handle.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    if val_idx is not None:
        x_val = to_chan_first(val_idx, fit_scaler=False)
        val_ds = TensorDataset(torch.from_numpy(x_val),
                               torch.from_numpy(y_encoded[val_idx].astype(np.int64)))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        val_ds = val_loader = None

    model = _CNNNet(c, n_classes, dropout).to(device)

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

    model.to('cpu').eval()
    clf = _CNNClassifier(model, label_encoder, scaler, label_encoder.classes_)
    clf.history_ = history
    return clf, train_time
