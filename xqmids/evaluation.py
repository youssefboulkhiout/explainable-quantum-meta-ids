"""Stages 6/6b: metrics, confusion matrices, ROC curves, per-class zero-day report and figures.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, cohen_kappa_score, confusion_matrix, roc_curve, auc

from ._env import HAS_SNS, sns
from .config import ALL_MODELS, META_MODELS, QCONTROL_MODELS, QUANTUM_MODELS
from .utils import _smooth, heatmap_df, sanitize_proba, savefig

# =============================================================================
# 6. EVALUATION (metrics + confusion matrix + ROC + feature importance)
# =============================================================================
def compute_metrics(y_true, y_pred, y_proba, n_classes):
    """Standard classification metrics (macro-averaged for multiclass)."""
    m = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Kappa":     cohen_kappa_score(y_true, y_pred),
    }
    try:
        present = np.unique(y_true)         # classes actually in this test split
        if len(present) < 2:
            m["AUC"] = np.nan               # AUC undefined with a single class
        elif len(present) == 2:
            m["AUC"] = roc_auc_score(y_true, y_proba[:, present[1]])
        else:
            # restrict to classes present in y_true and renormalize, since
            # some rare classes may be entirely absent from this split
            sub = y_proba[:, present]
            sub = sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
            m["AUC"] = roc_auc_score(y_true, sub, multi_class="ovr",
                                     average="macro", labels=present)
    except Exception:
        m["AUC"] = np.nan
    return m

  
    
def plot_confusion(y_true, y_pred, class_names, title, path, normalize=True):
    """Row-normalised confusion-matrix heatmap (annotated when <=12 classes)."""    
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    if normalize : cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    figsize = (max(5, 0.45 * len(class_names)), max(4, 0.35 * len(class_names)))
    fig, ax = plt.subplots(figsize=figsize)
    if HAS_SNS:
        sns.heatmap(cm, annot=len(class_names) <= 12, fmt=".2f" if normalize else "d",
                    cmap="Blues" if normalize else "icefire" , xticklabels=class_names,
                    yticklabels=class_names, cbar=True, ax=ax,
                    annot_kws={"fontsize": 8}  )
    else:
        im = ax.imshow(cm, cmap="Blues" if normalize else "icefire"); fig.colorbar(im)
        ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    savefig(fig, path)


def plot_roc(y_true, y_proba, class_names, title, path):
    """One-vs-rest ROC curves: a micro-average plus the 8 most frequent
    classes, each labelled with its AUC."""
    n_classes = y_proba.shape[1]          # always match the proba matrix
    y_proba = sanitize_proba(y_proba)     # defensive: reject any NaN/inf
    fig, ax = plt.subplots(figsize=(7, 6))
    y_bin = np.zeros((len(y_true), n_classes))
    y_bin[np.arange(len(y_true)), y_true] = 1
    # micro-average
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    ax.plot(fpr, tpr, "k--", lw=2, label=f"micro-avg (AUC={auc(fpr, tpr):.3f})")
    shown = np.argsort(-np.bincount(y_true, minlength=n_classes))[:8]
    for c in shown:
        if y_bin[:, c].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
        ax.plot(fpr, tpr, lw=1,
                label=f"{class_names[c][:18]} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    savefig(fig, path)


def plot_lgbm_importance(clf, feat_names, path):
    """Horizontal bar chart of the top-30 LightGBM features by split gain."""
    imp = clf.booster_.feature_importance(importance_type="gain")
    order = np.argsort(imp)[::-1][:30]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh([feat_names[i] for i in order][::-1], imp[order][::-1],
            color="#3070b3")
    ax.set_title("LightGBM feature importance (gain)")
    savefig(fig, path)


# ---- training-dynamics figures ----------------------------------------------
def plot_cnn_loss(history, ds_name, path):
    """Train vs validation cross-entropy per epoch for the CNN."""
    if not history or not history.get("train"):
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ep = np.arange(1, len(history["train"]) + 1)
    ax.plot(ep, history["train"], "-o", ms=3, label="train", color="#3070b3")
    ax.plot(ep, history["val"], "-s", ms=3, label="validation", color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title(f"{ds_name} -- CNN learning curves")
    ax.legend()
    savefig(fig, path)


def plot_meta_losses(meta_histories, ds_name, figdir):
    """(a) all meta-learners on one figure, (b) one small figure per learner."""
    if not meta_histories:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for algo, hist in meta_histories.items():
        if not hist:
            continue
        s = _smooth(hist)
        ax.plot(np.arange(len(s)), s, lw=1.4, label=algo)
    ax.set_xlabel("Meta-training episode"); ax.set_ylabel("Query loss (smoothed)")
    ax.set_title(f"{ds_name} -- meta-training loss curves")
    ax.set_yscale("log"); ax.legend(fontsize=8, ncol=2)
    savefig(fig, os.path.join(figdir, "loss_meta_all.png"))
    for algo, hist in meta_histories.items():
        if not hist:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(hist, lw=0.6, alpha=0.35, color="#3070b3")
        s = _smooth(hist)
        ax.plot(np.arange(len(s)), s, lw=1.8, color="#d62728", label="smoothed")
        ax.set_xlabel("Episode"); ax.set_ylabel("Loss")
        ax.set_title(f"{ds_name} -- {algo} meta-training loss"); ax.legend()
        savefig(fig, os.path.join(figdir, f"loss_meta_{algo}.png"))


# ---- computational-cost figure ----------------------------------------------
def plot_time_comparison(perf_df, ds_name, path):
    """Grouped bar chart (log y) of train / inference / XAI seconds per model."""
    cols = [c for c in ["Train_time_s", "Eval_time_s", "XAI_time_s"]
            if c in perf_df.columns]
    if not cols:
        return
    piv = perf_df.set_index("Model")[cols]
    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot(kind="bar", ax=ax, width=0.8)
    ax.set_yscale("log"); ax.set_ylabel("Seconds (log scale)")
    ax.set_title(f"{ds_name} -- training / inference / XAI time per model")
    ax.legend(["Training", "Evaluation", "XAI (SHAP+LIME)"][:len(cols)])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    savefig(fig, path)


# ---- zero-day statistics and figures ----------------------------------------
def zeroday_global_report(ds_name, models, X_te, y_te, zero_day, class_names):
    """Per-class detection statistics on the FULL test set for the held-out
    zero-day classes. Meta-learners saw them only through the K-shot global
    adaptation support; base learners saw them fully during training and thus
    act as an upper reference."""
    rows = []
    for name, (pp, _) in models.items():
        pred = np.argmax(pp(X_te), axis=1)
        for c in zero_day:
            mask = y_te == c
            if mask.sum() == 0:
                continue
            tp = int(np.sum(pred[mask] == c))
            fp = int(np.sum((pred == c) & ~mask))
            recall = tp / int(mask.sum())
            prec = tp / max(tp + fp, 1)
            f1 = 2 * prec * recall / max(prec + recall, 1e-12)
            rows.append({"Dataset": ds_name, "Model": name,
                         "Class": class_names[c], "Support": int(mask.sum()),
                         "DetectionRate": recall, "Precision": prec, "F1": f1,
                         "Regime": ("few-shot (K-shot support only)"
                                    if name in META_MODELS
                                    else "quantum/q-features (subsampled)"
                                    if name in QUANTUM_MODELS + QCONTROL_MODELS
                                    else "fully supervised (reference)")})
    return rows


def plot_zeroday_figures(zd_df, fs_df, ds_name, figdir):
    """Zero-day recall heatmap + few-shot vs zero-day episodic comparison."""
    if zd_df is not None and not zd_df.empty:
        piv = zd_df.pivot_table(index="Model", columns="Class",
                                values="DetectionRate")
        piv = piv.reindex([m for m in ALL_MODELS if m in piv.index])
        heatmap_df(piv, f"{ds_name} -- zero-day detection rate (recall)",
                   os.path.join(figdir, "zeroday_recall_heatmap.png"),
                   cmap="rocket" if HAS_SNS else "viridis")
    if fs_df is not None and not fs_df.empty and \
            "zeroday_episodic_accuracy" in fs_df.columns:
        sub = fs_df.set_index("Model")
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.2
        x = np.arange(len(sub))
        pairs = [("fewshot_episodic_accuracy", "Few-shot Acc", "#3070b3"),
                 ("zeroday_episodic_accuracy", "Zero-day Acc", "#d62728"),
                 ("fewshot_episodic_f1_macro", "Few-shot F1", "#7fb3d5"),
                 ("zeroday_episodic_f1_macro", "Zero-day F1", "#f1948a")]
        for j, (col, lab, color) in enumerate(pairs):
            if col in sub.columns:
                ax.bar(x + (j - 1.5) * width, sub[col].values, width,
                       label=lab, color=color)
        ax.set_xticks(x); ax.set_xticklabels(sub.index, rotation=45, ha="right")
        ax.set_ylim(0, 1.02); ax.set_ylabel("Episodic score")
        ax.set_title(f"{ds_name} -- few-shot vs zero-day episodic performance")
        ax.legend(fontsize=8)
        savefig(fig, os.path.join(figdir, "fewshot_vs_zeroday.png"))
