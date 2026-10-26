"""Stage 5A: supervised base learners (LightGBM, 1-D CNN).

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import time
import copy
from tqdm import tqdm
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, SEED
from .utils import sanitize_proba, set_seed

# =============================================================================
# 5A. BASE LEARNERS
# =============================================================================
def train_lightgbm(X_tr, y_tr, X_va, y_va):
    """Fit a LightGBM multiclass classifier with early stopping on the
    validation set. Returns the fitted classifier and wall-clock train time."""
    clf = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        subsample=0.9, colsample_bytree=0.9, class_weight=None,
        random_state=SEED, n_jobs=-1, verbose=-1)

    # progress bar ticks once per boosting round (not once per epoch)
    pbar = tqdm(total=clf.n_estimators, desc="\t\t\t[LightGBM]", leave=False)
    def tqdm_callback(env):
        """LightGBM callback: advance the progress bar one boosting round."""
        pbar.update(1)
    start_train = time.time()
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
            callbacks=[tqdm_callback, lgb.early_stopping(40, verbose=False),
                       lgb.log_evaluation(0)])

    training_time = time.time() - start_train
    return clf, training_time


def make_full_proba(clf, n_classes):
    """LightGBM only outputs columns for the classes it was trained on
    (rare classes are removed from the train set during balancing).
    Re-expand its probabilities to the full label space so every model
    exposes an (n, n_classes) predict_proba."""
    cls = np.asarray(clf.classes_, dtype=int)

    def pp(X):
        """Predict, then scatter the local-class probabilities back into the
        full (n, n_classes) label space (zeros for classes the model never saw)."""
        p = clf.predict_proba(np.asarray(X))
        if p.shape[1] == n_classes:
            return sanitize_proba(p)
        full = np.zeros((p.shape[0], n_classes), dtype=float)
        full[:, cls] = p
        return sanitize_proba(full)
    return pp


class CNN1D(nn.Module):
    """Simple 1-D CNN over the (standardized) feature vector."""
    def __init__(self, in_dim: int, n_classes: int):
        """Build a 2-conv feature extractor + 2-layer MLP classification head.
        AdaptiveAvgPool makes the head size independent of the input width."""
        super().__init__()
        # treat the feature vector as a length-F, single-channel "signal"
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),   # fixed-length output regardless of in_dim
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 4, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, n_classes),
        )

    def forward(self, x):                     # x: (B, F)
        """Add a channel dim, run the conv feature extractor, then classify."""
        x = x.unsqueeze(1)                    # (B, 1, F) add channel dim
        return self.classifier(self.features(x))


def train_cnn(X_tr, y_tr, X_va, y_va, n_classes, cfg):
    """Train the 1-D CNN with Adam + early stopping on validation loss.
    Keeps the best-val-loss weights (not necessarily the last epoch's)."""
    set_seed()
    model = CNN1D(X_tr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    crit = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X_tr, dtype=torch.float32)
    yt = torch.as_tensor(y_tr, dtype=torch.long)
    Xv = torch.as_tensor(X_va, dtype=torch.float32).to(DEVICE)
    yv = torch.as_tensor(y_va, dtype=torch.long).to(DEVICE)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"],
                                     shuffle=True, drop_last=False)
    best_state, best_loss, patience = None, np.inf, 0    # early-stopping state
    history = {"train": [], "val": []}
    start_train = time.time()
    for epoch in tqdm(range(cfg["epochs"]), desc="\t\t\t[CNN]", unit="epoch", leave=False):
        model.train()
        ep_losses = []
        for xb, yb in dl:                      # mini-batch SGD step
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vloss = crit(model(Xv), yv).item()
        history["train"].append(float(np.mean(ep_losses)))
        history["val"].append(float(vloss))
        if vloss < best_loss - 1e-4:            # improved -> checkpoint + reset patience
            best_loss, best_state, patience = vloss, copy.deepcopy(model.state_dict()), 0
        else:                                   # no improvement -> count toward patience
            patience += 1
            if patience >= cfg["patience"]:
                break                            # early stop
    if best_state is not None:
        model.load_state_dict(best_state)       # restore best checkpoint
    training_time = time.time() - start_train
    model.eval()
    return model, training_time, history


class TorchWrapper:
    """scikit-learn-style predict_proba wrapper around a torch module."""
    def __init__(self, model, n_classes):
        """Wrap a trained torch classifier; classes_ is the 0..n_classes-1 range."""
        self.model, self.classes_ = model, np.arange(n_classes)

    def predict_proba(self, X):
        """Batched, no-grad softmax probabilities as a NumPy (n, n_classes) array."""
        X = np.asarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):      # chunk to bound GPU memory
                xb = torch.as_tensor(X[i:i + 4096]).to(DEVICE)
                outs.append(F.softmax(self.model(xb), dim=1).cpu().numpy())
        return sanitize_proba(np.vstack(outs))

    def predict(self, X):
        """Hard class predictions = argmax of predict_proba."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
