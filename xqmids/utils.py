"""Small utilities: seeding, logging, JSON/figure saving, probability guard.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from ._env import HAS_SNS, sns
from .config import SEED

# =============================================================================
# UTILITIES
# =============================================================================
def set_seed(seed: int = SEED):
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> str:
    """Create `path` (and parents) if missing and return it, for chaining."""
    os.makedirs(path, exist_ok=True)
    return path


def log(msg: str):
    """Timestamped, immediately-flushed stdout logger used throughout."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(obj, path):
    """Dump `obj` to `path` as UTF-8 JSON (non-serialisable -> str())."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def savefig(fig, path):
    """Save a Matplotlib figure at 200 DPI with a tight bounding box, then
    close it to free memory (important given the many figures produced)."""
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def sanitize_proba(p):
    """Replace NaN/+-inf with 0 and renormalise each row to a valid
    probability distribution. A diverged meta-learner (or a softmax over
    inf/NaN logits) can emit non-finite probabilities; downstream sklearn
    metrics (roc_curve, roc_auc_score, ...) reject non-finite input, so every
    predict_proba is passed through this guard before use."""
    p = np.asarray(p, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    s = p.sum(axis=1, keepdims=True)
    # rows that summed to 0 (all-NaN) fall back to a uniform distribution
    p = np.where(s > 0, p / np.maximum(s, 1e-12), 1.0 / max(p.shape[1], 1))
    return p


def _smooth(v, frac=0.02):
    """Moving average with a window proportional to the sequence length."""
    v = np.asarray(v, dtype=float)
    w = max(1, int(len(v) * frac))
    if w <= 1 or len(v) < 3:
        return v
    return np.convolve(v, np.ones(w) / w, mode="valid")


def heatmap_df(df, title, path, fmt=".3f", cmap="viridis"):
    """Generic annotated heatmap of a DataFrame (auto-sized), used for most
    of the comparison/summary figures. Falls back to imshow without seaborn."""
    fig, ax = plt.subplots(figsize=(1.2 + 0.8 * df.shape[1],
                                    1.0 + 0.5 * df.shape[0]))
    if HAS_SNS:
        sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, ax=ax,
                    cbar=True, linewidths=0.4)
    else:
        im = ax.imshow(df.values, cmap=cmap); fig.colorbar(im)
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=45)
        ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
        
    ax.tick_params(axis="x", labelrotation=45)
    ax.set_title(title)
    savefig(fig, path)
