"""
=============================================================================
 Explainable Quantum-Enhanced Meta-Learning for Few-Shot and Zero-Day 
 Intrusion Detection in IoT Networks: -- A Unified Benchmark Framework
=============================================================================
 Datasets (sequential) : RT-IoT2022, UNSW-NB15, ASEADOS_SDN_IoT
 Base learners         : LightGBM, CNN (1-D)
 Meta-learners         : Reptile, MAML, FOMAML, Meta-SGD, ANIL, BOIL, ProtoNet
 Quantum (Level A)     : QSVC (quantum-kernel SVM), VQC (variational quantum
                         classifier), QNN (data re-uploading quantum neural
                         network), HQCNN (hybrid quantum-classical NN)
                         + classical controls SVM-q / MLP-q on the same q features
 Quantum (Level B)     : hybrid quantum prototypical networks, meta-learned
                         episodically: QSVC-PN (trainable quantum-kernel
                         ProtoNet), VQC-PN, QNN-PN, HQCNN-PN
                         + classical control MLP-PN on the same q features
 Explainability        : SHAP + LIME with 8 explanation-quality metrics
                         (Fidelity, Infidelity, Sparsity, Stability,
                          Comprehensiveness, Sufficiency,
                          Feature Agreement, Rank Correlation)

 Pipeline stages
 ---------------
 1. Data loading & cleaning (auto label detection, drop irrelevant features,
    remove NaN/inf and duplicates)
 2. Split 70/10/20 (train/val/test, stratified) + StandardScaler
 3. Feature selection: SelectKBest(f_classif), k = 30, fitted on TRAIN only
 4. Class balancing on TRAIN only:
       - remove classes with  < MIN_CLASS_SIZE_TO_KEEP samples
       - SMOTE classes in    [MIN_CLASS_SIZE_TO_KEEP, TARGET_MIN_CLASS_SIZE)
         up to TARGET_MIN_CLASS_SIZE
       - cap classes with    > MAX_CLASS_SIZE samples at MAX_CLASS_SIZE
 3Q. Quantum-only extra reduction: top-q (q = 4-8) of the 30 features,
    mapped to rotation angles in [0, pi]
 5. Model training (base + 7 meta-learners + 4 quantum models; meta-learners are trained
    episodically N-way/K-shot, then globally adapted with a small support
    set for full test-set evaluation; optional zero-day held-out classes)
 6. Evaluation: Accuracy, AUC, Precision, Recall, F1, Cohen's Kappa
    + confusion matrices, ROC curves, feature importance plots
 6b. Zero-day per-class detection statistics on the full test set
 6c. UNIFIED zero-day (open-set / novelty) detection for ALL models
     (both base learners AND meta-learners): every model is re-fit /
     re-adapted on the KNOWN classes only, then scored with model-agnostic
     novelty scores (MSP / normalised entropy / top-2 margin). Reported with
     AUROC, AUPR, FPR@95%TPR, detection-rate @ fixed FPR, and best-F1.
 7. SHAP & LIME explanations + 8 quality metrics
 8. Reporting: per-dataset CSVs, heatmaps, beeswarm/waterfall/decision
    plots, cross-dataset / cross-model / cross-method comparison figures

 Usage
 -----
   python xai_meta_ids.py                       # all datasets
   python xai_meta_ids.py --datasets RT-IoT2022 UNSW-NB15
   python xai_meta_ids.py --quick               # fast smoke test
   python xai_meta_ids.py --episodes 500        # more meta-training episodes
   python xai_meta_ids.py --qubits 6            # quantum models with 6 qubits
   python xai_meta_ids.py --eval-shots 1000     # finite-shot quantum evaluation
   python xai_meta_ids.py --no-quantum          # original classical pipeline
   python xai_meta_ids.py --level B             # only Level-B quantum models
   python xai_meta_ids.py --level A             # only Level-A quantum models

 Requirements
 ------------
   pip install numpy pandas scipy scikit-learn imbalanced-learn lightgbm \
               torch shap lime matplotlib seaborn tqdm pennylane
=============================================================================
"""

import os
import json
import time
import copy
import random
import warnings
import argparse
from tqdm import tqdm
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.9)
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, cohen_kappa_score,
                             confusion_matrix, roc_curve, auc)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import lightgbm as lgb

import torch
import torch.nn as nn
import torch.nn.functional as F

import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "output_dir": "outputs",
    "test_size": 0.20,          # 70 / 10 / 20
    "val_size":  0.10,
    "k_best": 30,               # SelectKBest features
    "min_class_size_to_keep": 50,
    "target_min_class_size": 10000,
    "max_class_size": 50000,
    "max_rows": 200000,         # stratified cap on raw rows (None = no cap)
    "min_class_presplit": 10,   # needed for stratified 70/10/20 split

    "cnn": {
        "epochs": 40, 
        "batch_size": 512, 
        "lr": 1e-3, 
        "patience": 5,
    },
    
    "meta": {
        "episodes": 500,        # meta-training episodes (more -> better convergence)
        "meta_batch": 4,        # tasks per meta-update (more -> stabler gradients)
        "n_way": 5,             # capped at number of available classes
        "k_shot": 20,           # more support examples per class per episode
        "q_query": 15,          # more query examples -> less noisy meta-gradient
        "inner_steps": 5,       # more inner-loop adaptation steps
        "inner_lr": 0.01,      # 0.001/0.01
        "meta_lr": 1e-3,        # lower, steadier outer LR for the longer run 5e-2/1e-3
        "reptile_lr": 0.05,     # inner Adam lr replaced by SGD lr 0.5/0.05
        "reptile_eps": 0.5,     # meta-interpolation (linearly annealed)
        "hidden": (128, 64),    # larger network capacity
        "adapt_shot": 20,       # support per class for global adaptation
        "adapt_steps": 50,      # fine-tuning steps for global adaptation 50/60
        "zero_day": True,       # hold out rare attack classes from meta-train
        "zero_day_n_classes": 2,
        "eval_episodes": 200,   # episodic few-shot / zero-day evaluation
    },

    "xai": {
        "n_background": 100,    # background samples for SHAP / LIME
        "n_explain": 40,        # instances explained per model/method
        "kernel_nsamples": 150, # SHAP KernelExplainer budget
        "lime_num_samples": 1000,
        "top_k_list": [5, 10, 15],
        "stability_instances": 5,
        "stability_perturbs": 3,
        "stability_sigma": 0.05,
        "infidelity_perturbs": 20,
        "infidelity_sigma": 0.3,
    },

    # ---- unified zero-day (open-set / novelty) detection (pipeline stage 6C) --
    # Settings that control the model-agnostic "is this an unseen attack?"
    # detector applied identically to every base learner and meta-learner.
    "zeroday_detect": {
        "enabled": True,                # master on/off switch for stage 6C
        "score_methods": ["msp", "entropy", "margin"],  # novelty scores to test
        "target_fprs": [0.05, 0.10],    # report detection rate (TPR) at these FPRs
        "retrain_base_known_only": True,  # re-fit LightGBM/CNN WITHOUT zero-day
        "max_test_known": 20000,        # cap known negatives when scoring (None=all)
        "retrain_quantum_known_only": True,  # re-fit quantum models WITHOUT zero-day
    },

    # ---- Level-A quantum extension (pipeline stages 3Q / 5C) ---------------
    # Exact statevector simulation with PennyLane on CPU (no GPU, no QPU).
    "quantum": {
        "enabled": True,
        "n_qubits": 4,              # q: top-q of the 30 features (4-8); raised
                                    # automatically to ceil(log2 n_classes) so
                                    # the VQC basis-state readout covers all classes
        "device": "default.qubit",  # PennyLane simulator
        "feature_map": "zz",        # QSVC kernel encoding: "zz" or "angle"
        "fmap_reps": 2,             # repetitions of the ZZ feature map
        "kernel_bandwidth": 0.25,   # angle scaling inside the QSVC feature map
                                    # (Shaydulin & Wild 2022). 1.0 = raw ZZ map,
                                    # which is over-expressive (near-diagonal
                                    # kernel); smaller -> smoother kernel
        "qsvc_per_class": 60,       # QSVC training rows per class (kernel O(n^2))
        "qsvc_C": 1.0,              # SVM regularisation (QSVC and SVM-q)
        "train_per_class": 200,     # VQC/QNN/HQCNN training rows per class
        "val_per_class": 40,        # validation rows per class (early stopping)
        "layers": 3,                # StronglyEntanglingLayers depth
        "epochs": 30,
        "batch_size": 32,
        "lr": 0.02,
        "patience": 6,
        "eval_shots": None,         # None = exact; e.g. 1000 = emulate finite-shot
                                    # measurement noise in the evaluation stage
        "classical_controls": True, # add SVM-q / MLP-q on identical q features
    },

    # ---- Level-B hybrid quantum prototypical networks (stage 5D) -----------
    # Uses the same q features / angle mapping as Level A (cfg["quantum"]).
    "quantum_b": {
        "enabled": True,
        "device": "default.qubit",
        "embed_dim": 16,            # d: embedding dimension (8-32)
        "encoding": "angle",        # "angle" or "amplitude" (VQC-PN / HQCNN-PN)
        "layers": 3,                # StronglyEntanglingLayers depth
        "fmap_reps": 2,             # QSVC-PN: data layers (trainable layer between)
        "kernel_bandwidth_init": 0.25,  # QSVC-PN: initial per-feature bandwidth
        "tau_init": 5.0,            # initial inverse temperature of softmax(-tau*d)
        "episodes": 300,            # meta-training episodes
        "meta_batch": 2,            # episodes per meta-update
        "n_way": 5,                 # capped at number of available classes
        "k_shot": 20,               # K support shots per class (figure: K = 20)
        "q_query": 15,
        "lr": 0.01,
        "adapt_shot": 20,           # support per class for global prototypes
        "eval_episodes": 200,       # episodic few-shot / zero-day evaluation
        "eval_shots": None,         # None = exact; int = finite-shot evaluation
        "classical_control": True,  # MLP-PN on the same q features & episodes
    },

    # bootstrap resamples for 95% CIs (Accuracy, Macro-F1, zero-day AUROC);
    # these are the error bars of the step-8 comparison figure. 0 = off
    "bootstrap": 200,
}

# Per-dataset configuration: edit "path" entries to match your local files.
DATASETS = {
    "ASEADOS_SDN_IoT": {
        "path": "data/ASEADOS_SDN_IoT.csv",
        "label_candidates": ["label", "class", "attack_type", "category"],
        "drop": ["id", "flow_id", ],
    },
    "RT-IoT2022": {
        "path": "data/RT-IoT2022.csv",
        "label_candidates": ["attack_type", "label", "class"],
        "drop": ["no.", "no", "id", "unnamed: 0"],
    },
    "UNSW-NB15": {
        "path": "data/UNSW-NB15.csv",
        "label_candidates": ["attack_cat"],
        "drop": ["id", "label", ],
    },
}

GENERIC_DROP_EXACT = {
    "id", "no", "no.", "index", "unnamed: 0", "flow id", "flow_id",
}

BASE_MODELS = ["LightGBM", "CNN"]
META_MODELS = ["Reptile", "MAML", "FOMAML", "Meta-SGD", "ANIL", "BOIL", "ProtoNet"] #
QUANTUM_MODELS = ["QSVC", "VQC", "QNN", "HQCNN"]          # Level-A QML
# Level-B hybrid quantum prototypical networks (meta-learned quantum embeddings)
QUANTUM_B_MODELS = ["QSVC-PN", "VQC-PN", "QNN-PN", "HQCNN-PN"]
# classical controls on the same q features (Level A: SVM-q, MLP-q;
# Level B: MLP-PN = classical ProtoNet with an MLP embedding)
QCONTROL_MODELS = ["SVM-q", "MLP-q", "MLP-PN"]
ALL_MODELS = (BASE_MODELS + META_MODELS + QUANTUM_MODELS + QUANTUM_B_MODELS
              + QCONTROL_MODELS)


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


# =============================================================================
# 1. DATA LOADING AND CLEANING
# =============================================================================
def detect_label_column(df: pd.DataFrame, candidates) -> str:
    """Auto-detect the label column (dataset candidates first, then fuzzy)."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for key in ["attack_cat", "attack_type", "label", "attack", "class", "category"]:
        for c in df.columns:
            if key in c.lower():
                return c
    return df.columns[-1]  # fallback: last column


def load_and_clean(name: str, dcfg: dict, cfg: dict):
    """Load a CSV, detect label, drop irrelevant columns, clean rows."""
    path = dcfg["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[{name}] file not found: {path} -- edit DATASETS['{name}']['path']")
    log(f"[{name}] loading {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    label_col = detect_label_column(df, dcfg.get("label_candidates", []))
    log(f"[{name}] detected label column: '{label_col}'")

    # ---- drop irrelevant columns -------------------------------------------
    drop_set = {c.lower() for c in dcfg.get("drop", [])} | GENERIC_DROP_EXACT
    to_drop = [c for c in df.columns
               if c != label_col and c.lower().strip() in drop_set]
    df = df.drop(columns=to_drop, errors="ignore")
    if to_drop:
        log(f"[{name}] dropped {len(to_drop)} irrelevant columns")

    # ---- clean labels -------------------------------------------------------
    y_raw = df[label_col].astype(str).str.strip()
    X = df.drop(columns=[label_col])

    # ---- encode / drop object feature columns ------------------------------
    obj_cols = X.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        nuniq = X[c].nunique(dropna=True)
        if nuniq > 50:                       # identifier-like -> drop
            X = X.drop(columns=[c])
        else:                                # low-cardinality categorical
            X[c] = pd.factorize(X[c].astype(str))[0]

    # ---- remove inf / NaN / duplicates --------------------------------------
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1)
    X, y_raw = X.loc[mask], y_raw.loc[mask]

    dup_mask = ~pd.concat([X, y_raw.rename("__y__")], axis=1).duplicated()
    X, y_raw = X.loc[dup_mask], y_raw.loc[dup_mask]

    # drop constant columns
    nunique = X.nunique()
    X = X[nunique[nunique > 1].index]

    # ---- drop ultra-rare classes (stratified split feasibility) -------------
    vc = y_raw.value_counts()
    keep = vc[vc >= cfg["min_class_presplit"]].index
    sel = y_raw.isin(keep)
    X, y_raw = X.loc[sel], y_raw.loc[sel]

    # ---- optional stratified row cap ----------------------------------------
    if cfg["max_rows"] and len(X) > cfg["max_rows"]:
        frac = cfg["max_rows"] / len(X)
        idx, _ = train_test_split(np.arange(len(X)), train_size=frac,
                                  stratify=y_raw.values, random_state=SEED)
        X, y_raw = X.iloc[idx], y_raw.iloc[idx]
        log(f"[{name}] capped to {len(X)} rows (stratified)")
        # re-drop classes that fell below the pre-split minimum after capping
        vc = y_raw.value_counts()
        keep = vc[vc >= cfg["min_class_presplit"]].index
        if len(keep) < len(vc):
            sel = y_raw.isin(keep)
            log(f"[{name}] dropped {int((~sel).sum())} rows of "
                f"{len(vc) - len(keep)} classes made too rare by the cap")
            X, y_raw = X.loc[sel], y_raw.loc[sel]

    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)
    log(f"[{name}] cleaned shape: {X.shape}, classes: {len(le.classes_)}")
    return X.reset_index(drop=True), y, le


# =============================================================================
# 2. SPLITTING AND SCALING (70 / 10 / 20)
# =============================================================================
def split_and_scale(X: pd.DataFrame, y: np.ndarray, cfg: dict):
    """Stratified 70/10/20 train/val/test split followed by a StandardScaler
    fitted on TRAIN only (then applied to val/test to avoid leakage). Rare
    classes that cannot survive two stratified splits are dropped first."""
    # defensive guard: every class must survive TWO stratified splits
    min_needed = max(int(cfg.get("min_class_presplit", 10)), 10)
    counts = Counter(y)
    keep = np.array(sorted(c for c, n in counts.items() if n >= min_needed))
    if len(keep) < len(counts):
        mask = np.isin(y, keep)
        log(f"split: removed {len(counts) - len(keep)} classes with "
            f"< {min_needed} samples ({int((~mask).sum())} rows)")
        X, y = X[mask], y[mask]

    X_tmp, X_te, y_tmp, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=SEED)
    val_frac = cfg["val_size"] / (1.0 - cfg["test_size"])   # 0.10 of total
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tmp, y_tmp, test_size=val_frac, stratify=y_tmp, random_state=SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)
    log(f"split -> train {X_tr.shape[0]} / val {X_va.shape[0]} / test {X_te.shape[0]}")
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), scaler


# =============================================================================
# 3. FEATURE SELECTION (SelectKBest, f_classif, fitted on TRAIN only)
# =============================================================================
def select_features(X_tr, y_tr, X_va, X_te, feat_names, k: int):
    """Univariate ANOVA F-test feature selection (SelectKBest), fitted on
    TRAIN only, then applied to all three splits. Returns the transformed
    matrices, the kept feature names, and the fitted selector."""
    k = min(k, X_tr.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    idx = selector.get_support(indices=True)
    names = [feat_names[i] for i in idx]
    log(f"SelectKBest kept {k} features")
    return (selector.transform(X_tr), selector.transform(X_va),
            selector.transform(X_te), names, selector)


# =============================================================================
# 4. CLASS BALANCING (train set only)
# =============================================================================
def balance_classes(X_tr, y_tr, cfg: dict):
    """Three-stage TRAIN-only class balancing: (a) drop classes below
    MIN_CLASS_SIZE_TO_KEEP, (b) SMOTE mid-size classes up to
    TARGET_MIN_CLASS_SIZE, (c) random-undersample majority classes down to
    MAX_CLASS_SIZE. Returns the balanced (X, y) and the list of kept classes
    (used to keep val/test consistent)."""
    mn, tgt, mx = (cfg["min_class_size_to_keep"],
                   cfg["target_min_class_size"], cfg["max_class_size"])
    counts = Counter(y_tr)

    # (a) remove classes below MIN_CLASS_SIZE_TO_KEEP
    keep_classes = sorted([c for c, n in counts.items() if n >= mn])
    mask = np.isin(y_tr, keep_classes)
    X_tr, y_tr = X_tr[mask], y_tr[mask]
    counts = Counter(y_tr)
    removed = [c for c in counts if c not in keep_classes]
    log(f"balancing: kept {len(keep_classes)} classes "
        f"(removed {len(set(Counter(y_tr)) ^ set(keep_classes)) if removed else 0} rare ones)")

    # (b) SMOTE mid-size classes up to TARGET_MIN_CLASS_SIZE
    smote_strategy = {c: tgt for c, n in counts.items() if mn <= n < tgt}
    if smote_strategy:
        k_nb = max(1, min(5, min(counts[c] for c in smote_strategy) - 1))
        try:
            X_tr, y_tr = SMOTE(sampling_strategy=smote_strategy,
                               k_neighbors=k_nb,
                               random_state=SEED).fit_resample(X_tr, y_tr)
            log(f"balancing: SMOTE applied to {len(smote_strategy)} classes -> {tgt}")
        except Exception as e:
            log(f"balancing: SMOTE skipped ({e})")

    # (c) cap majority classes at MAX_CLASS_SIZE
    counts = Counter(y_tr)
    under_strategy = {c: mx for c, n in counts.items() if n > mx}
    if under_strategy:
        X_tr, y_tr = RandomUnderSampler(sampling_strategy=under_strategy,
                                        random_state=SEED).fit_resample(X_tr, y_tr)
        log(f"balancing: capped {len(under_strategy)} classes at {mx}")

    log(f"balanced train size: {X_tr.shape[0]}  "
        f"class dist: {dict(sorted(Counter(y_tr).items()))}")
    return X_tr, y_tr, sorted(keep_classes)


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


# =============================================================================
# 5B. META-LEARNING FRAMEWORK (functional MLP, 7 algorithms)
# =============================================================================
# Instead of an nn.Module, meta-parameters are plain tensors in an OrderedDict.
# This "functional" style lets each algorithm clone/perturb/partially-update
# the weights per-episode without fighting PyTorch's module/optimizer state.
BODY_KEYS = ["w1", "b1", "w2", "b2"]     # shared feature-extractor weights
HEAD_KEYS = ["head_w", "head_b"]         # final classification layer


def init_meta_params(in_dim, n_classes, hidden=(128, 64)):
    """Random-init a 2-hidden-layer MLP (body) + linear head as a flat dict
    of leaf tensors, so it can be adapted/cloned functionally per episode."""
    h1, h2 = hidden
    def lin(o, i):
        """Kaiming-uniform weight (o x i) + zero bias, both requiring grad."""
        w = torch.empty(o, i, device=DEVICE)
        nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        b = torch.zeros(o, device=DEVICE)
        return w.requires_grad_(), b.requires_grad_()
    p = OrderedDict()
    p["w1"], p["b1"] = lin(h1, in_dim)
    p["w2"], p["b2"] = lin(h2, h1)
    p["head_w"], p["head_b"] = lin(n_classes, h2)
    return p


def body_forward(x, p):
    """Feature extractor only (used by ProtoNet prototypes / ANIL / BOIL)."""
    x = F.relu(F.linear(x, p["w1"], p["b1"]))
    x = F.relu(F.linear(x, p["w2"], p["b2"]))
    return x                                     # embedding


def full_forward(x, p):
    """Body + classification head -> class logits."""
    return F.linear(body_forward(x, p), p["head_w"], p["head_b"])


def clone_params(p, detach=True):
    """Deep-copy the parameter dict. detach=True starts a fresh autograd
    graph (used for each episode's inner-loop adaptation)."""
    return OrderedDict((k, (v.detach().clone().requires_grad_() if detach
                            else v) ) for k, v in p.items())


def sample_episode(X, y, class_pool, n_way, k_shot, q_query, rng):
    """N-way K-shot episode using ORIGINAL label ids (shared global head)."""
    classes = rng.choice(class_pool, size=min(n_way, len(class_pool)),
                         replace=False)
    sx, sy, qx, qy = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        # sample support (k_shot) + query (q_query) examples for this class;
        # allow sampling with replacement if the class doesn't have enough rows
        take = rng.choice(idx, size=min(len(idx), k_shot + q_query),
                          replace=len(idx) < k_shot + q_query)
        sx.append(X[take[:k_shot]]);        sy.append(np.full(k_shot, c))
        qx.append(X[take[k_shot:]]);        qy.append(np.full(len(take) - k_shot, c))
    to_t = lambda a, dt: torch.as_tensor(np.concatenate(a), dtype=dt, device=DEVICE)
    return (to_t(sx, torch.float32), to_t(sy, torch.long),
            to_t(qx, torch.float32), to_t(qy, torch.long))


def inner_adapt(params, sx, sy, update_keys, inner_lr, steps,
                second_order=False, lr_dict=None):
    """Gradient-based inner loop; returns adapted (fast) parameters.
    - update_keys: which parameter tensors get adapted (all/body-only/head-only)
      -> this is what distinguishes MAML (all) from ANIL (head) / BOIL (body).
    - second_order=True keeps the graph through the inner steps (true MAML),
      False detaches it each step (first-order approximation, e.g. FOMAML).
    - lr_dict, if given, supplies a PER-PARAMETER learned step size (Meta-SGD)
      instead of a single scalar inner_lr."""
    fast = OrderedDict(params)
    for _ in range(steps):
        loss = F.cross_entropy(full_forward(sx, fast), sy)
        grads = torch.autograd.grad(loss, [fast[k] for k in update_keys],
                                    create_graph=second_order)
        new = OrderedDict(fast)
        for k, g in zip(update_keys, grads):
            step = (lr_dict[k] * g) if lr_dict is not None else inner_lr * g
            new[k] = fast[k] - step
        fast = new
    return fast


def proto_logits(sx, sy, qx, params):
    """ProtoNet logits = negative squared Euclidean distance to prototypes.
    Prototypes are the per-class mean embedding of the support set; query
    points are classified by nearest prototype (metric-based, no gradient
    adaptation at inference time)."""
    emb_s, emb_q = body_forward(sx, params), body_forward(qx, params)
    classes = torch.unique(sy)
    protos = torch.stack([emb_s[sy == c].mean(0) for c in classes])
    d = torch.cdist(emb_q, protos) ** 2
    return -d, classes


def meta_train(algo, X_tr, y_tr, class_pool, in_dim, n_classes, mcfg):
    """Episodic meta-training for one algorithm; returns meta-parameters.

    All 7 algorithms share the same episodic sampling and the same MLP body,
    they only differ in (a) which parameters get inner-loop-adapted per task,
    (b) whether the inner loop keeps second-order gradients, and (c) how the
    outer/meta update is applied (Adam meta-gradient vs. Reptile's direct
    parameter interpolation)."""
    set_seed()
    rng = np.random.RandomState(SEED)
    params = init_meta_params(in_dim, n_classes, mcfg["hidden"])
    n_way = min(mcfg["n_way"], len(class_pool))
    ks, qq, steps, ilr = (mcfg["k_shot"], mcfg["q_query"],
                          mcfg["inner_steps"], mcfg["inner_lr"])

    # ---- per-algorithm choice of (a) adaptable params (b) 2nd-order grads ----
    if algo == "ANIL":
        upd_keys, second = HEAD_KEYS, False       # adapt head only, first-order
    elif algo == "BOIL":
        upd_keys, second = BODY_KEYS, False        # adapt body only, first-order
    elif algo == "MAML":
        upd_keys, second = list(params.keys()), True   # adapt everything, true 2nd-order
    elif algo in ("FOMAML", "Meta-SGD", "Reptile"):
        # all adapt every parameter; Meta-SGD additionally learns per-param LR
        upd_keys, second = list(params.keys()), (algo == "Meta-SGD")
    else:                                        # ProtoNet
        upd_keys, second = BODY_KEYS, False        # no gradient inner loop at all

    lr_dict = None
    trainables = list(params.values())
    if algo == "Meta-SGD":
        # learnable per-parameter inner learning rates (in addition to the
        # parameters themselves) -- this is what makes Meta-SGD "Meta-SGD"
        lr_dict = OrderedDict((k, torch.full_like(v, ilr).requires_grad_())
                              for k, v in params.items())
        trainables += list(lr_dict.values())

    if algo != "Reptile":
        # Reptile doesn't use an outer optimizer/meta-gradient at all; it
        # directly interpolates params toward the post-adaptation weights.
        opt = torch.optim.Adam(trainables, lr=mcfg["meta_lr"])
    
    loss = 0.0    
    history = []                       # per-episode training loss
    start_train = time.time()
    for ep in tqdm(range(mcfg["episodes"]), desc=f"\t\t\t[{algo}]", unit="episode", leave=False):
    #for ep in range(mcfg["episodes"]):
        if algo == "Reptile":                                     # ---- Reptile
            # linearly anneal the meta step size from reptile_eps -> 0
            eps = mcfg["reptile_eps"] * (1 - ep / mcfg["episodes"])
            ep_losses = []
            for _ in range(mcfg["meta_batch"]):
                sx, sy, qx, qy = sample_episode(X_tr, y_tr, class_pool,
                                                n_way, ks, qq, rng)
                # Reptile trains on support+query jointly (no separate query loss)
                tx = torch.cat([sx, qx]); ty = torch.cat([sy, qy])
                fast = clone_params(params)
                iopt = torch.optim.SGD(fast.values(), lr=mcfg["reptile_lr"])
                for _ in range(steps):              # k SGD steps on this task
                    iopt.zero_grad()
                    l_task = F.cross_entropy(full_forward(tx, fast), ty)
                    l_task.backward()
                    iopt.step()
                ep_losses.append(float(l_task))
                # meta-update: move params a small step toward the task-adapted
                # weights (Reptile's first-order approximation of MAML)
                with torch.no_grad():
                    for k in params:
                        params[k] += (eps / mcfg["meta_batch"]) * \
                                     (fast[k].detach() - params[k])
            history.append(float(np.mean(ep_losses)))
            continue

        opt.zero_grad()
        meta_loss = 0.0
        for _ in range(mcfg["meta_batch"]):         # average meta-gradient over a batch of tasks
            sx, sy, qx, qy = sample_episode(X_tr, y_tr, class_pool,
                                            n_way, ks, qq, rng)
            if algo == "ProtoNet":                                # ---- ProtoNet
                # metric-based: no inner-loop gradient steps, just compute
                # prototypes from the support set and classify the query set
                logits, classes = proto_logits(sx, sy, qx, params)
                y_map = torch.searchsorted(classes, qy)  # map global ids -> episode-local ids
                task_loss = F.cross_entropy(logits, y_map)
            else:                                                 # gradient-based
                if algo == "FOMAML":
                    # first-order MAML: adapt on the support set, then compute
                    # the query loss and its gradient w.r.t. the ADAPTED params,
                    # but manually copy that gradient onto the ORIGINAL params
                    # (skipping backprop through the inner-loop steps entirely)
                    fast = inner_adapt(params, sx, sy, upd_keys, ilr, steps,
                                       second_order=False)
                    fast = OrderedDict((k, v.detach().requires_grad_()
                                        if k in upd_keys else v)
                                       for k, v in fast.items())
                    task_loss = F.cross_entropy(full_forward(qx, fast), qy)
                    grads = torch.autograd.grad(task_loss,
                                                [fast[k] for k in upd_keys])
                    for k, g in zip(upd_keys, grads):   # first-order transport
                        g = g / mcfg["meta_batch"]
                        params[k].grad = (params[k].grad + g
                                          if params[k].grad is not None else g.clone())
                    meta_loss += task_loss.item()
                    continue
                # MAML (true 2nd-order), ANIL, BOIL, Meta-SGD all funnel through
                # here: adapt on support set (fast weights), evaluate query loss
                # on the fast weights, and let autograd backprop through the
                # whole inner loop into the ORIGINAL params (opt.step() below).
                fast = inner_adapt(params, sx, sy, upd_keys, ilr, steps,
                                   second_order=second, lr_dict=lr_dict)
                task_loss = F.cross_entropy(full_forward(qx, fast), qy)
            (task_loss / mcfg["meta_batch"]).backward()
            meta_loss += float(task_loss)
        # clip the meta-gradient to keep the outer update from exploding into
        # NaN weights (a common cause of non-finite softmax outputs later)
        torch.nn.utils.clip_grad_norm_(trainables, max_norm=5.0)
        opt.step()                                  # single meta-update per episode
        loss = meta_loss / mcfg['meta_batch']
        history.append(float(loss))
    
    training_time = time.time() - start_train
    #log(f"    [{algo}] meta-loss={loss:.4f}")
    return params, lr_dict, training_time, history


class MetaWrapper:
    """Globally-adapted meta-model exposing predict_proba on the full test set."""
    def __init__(self, algo, params, n_classes, protos=None, proto_classes=None):
        """Hold the (globally-adapted) meta-parameters; for ProtoNet also hold
        the class prototypes and their global class ids."""
        self.algo, self.params, self.classes_ = algo, params, np.arange(n_classes)
        self.protos, self.proto_classes = protos, proto_classes

    def _logits(self, xb):
        """Per-class logits: nearest-prototype distances for ProtoNet, else the
        MLP body+head forward pass. Absent ProtoNet classes get -1e9."""
        if self.algo == "ProtoNet":
            emb = body_forward(xb, self.params)
            d = torch.cdist(emb, self.protos) ** 2
            logits = torch.full((xb.shape[0], len(self.classes_)), -1e9,
                                device=DEVICE)
            logits[:, self.proto_classes] = -d
            return logits
        return full_forward(xb, self.params)

    def predict_proba(self, X):
        """Batched, no-grad softmax over the full label space, as (n, n_classes)."""
        X = np.asarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):      # chunk to bound GPU memory
                xb = torch.as_tensor(X[i:i + 4096]).to(DEVICE)
                outs.append(F.softmax(self._logits(xb), dim=1).cpu().numpy())
        return sanitize_proba(np.vstack(outs))

    def predict(self, X):
        """Hard class predictions = argmax of predict_proba."""
        return np.argmax(self.predict_proba(X), axis=1)


def global_adapt(algo, params, lr_dict, X_tr, y_tr, n_classes, mcfg):
    """Few-shot global adaptation: small support set over ALL train classes.

    After meta-training (which only ever sees n_way classes per episode),
    this step adapts the meta-learned initialization to the FULL label space
    using just `adapt_shot` examples per class, producing the model that is
    actually evaluated on the full test set (see run_dataset's evaluation
    stage)."""
    rng = np.random.RandomState(SEED)
    sx, sy = [], []
    for c in np.unique(y_tr):
        idx = np.where(y_tr == c)[0]
        take = rng.choice(idx, size=min(len(idx), mcfg["adapt_shot"]),
                          replace=False)
        sx.append(X_tr[take]); sy.append(np.full(len(take), c))
    sx = torch.as_tensor(np.concatenate(sx), dtype=torch.float32, device=DEVICE)
    sy = torch.as_tensor(np.concatenate(sy), dtype=torch.long, device=DEVICE)

    if algo == "ProtoNet":
        # no gradient steps: just recompute prototypes over ALL classes
        with torch.no_grad():
            emb = body_forward(sx, params)
            classes = torch.unique(sy)
            protos = torch.stack([emb[sy == c].mean(0) for c in classes])
        return MetaWrapper(algo, clone_params(params), n_classes,
                           protos=protos, proto_classes=classes.cpu().numpy())

    # gradient-based algorithms: fine-tune the relevant parameter subset with
    # plain SGD for `adapt_steps` steps on the global support set
    upd_keys = (HEAD_KEYS if algo == "ANIL"
                else BODY_KEYS if algo == "BOIL"
                else list(params.keys()))
    fast = clone_params(params)
    opt = torch.optim.SGD([fast[k] for k in upd_keys], lr=mcfg["inner_lr"])
    for _ in range(mcfg["adapt_steps"]):
        opt.zero_grad()
        F.cross_entropy(full_forward(sx, fast), sy).backward()
        opt.step()
    for v in fast.values():
        v.requires_grad_(False)             # freeze for inference
    return MetaWrapper(algo, fast, n_classes)


def episodic_eval(algo, params, lr_dict, X_te, y_te, class_pool, mcfg,
                  tag="fewshot"):
    """Episodic N-way K-shot evaluation (few-shot / zero-day accuracy & F1).

    Unlike global_adapt (adapt once on train, evaluate on the whole test
    set), this simulates the "true" few-shot setting: for each of
    `eval_episodes` episodes, sample a fresh support/query split from the
    TEST classes in `class_pool`, adapt from the meta-parameters using only
    that episode's support set, then score on its query set. `tag` is
    "fewshot" for meta_pool classes or "zeroday" for held-out classes."""
    rng = np.random.RandomState(SEED + 7)
    # only keep classes with enough test rows to form a support+query episode
    pool = [c for c in class_pool
            if np.sum(y_te == c) >= mcfg["k_shot"] + 2]
    if len(pool) < 2:
        return {}
    n_way = min(mcfg["n_way"], len(pool))
    accs, f1s = [], []
    for _ in range(mcfg["eval_episodes"]):
        sx, sy, qx, qy = sample_episode(X_te, y_te, np.array(pool), n_way,
                                        mcfg["k_shot"], mcfg["q_query"], rng)
        if algo == "ProtoNet":
            with torch.no_grad():
                logits, classes = proto_logits(sx, sy, qx, params)
                pred = classes[logits.argmax(1)]
        else:
            upd_keys = (HEAD_KEYS if algo == "ANIL"
                        else BODY_KEYS if algo == "BOIL"
                        else list(params.keys()))
            fast = clone_params(params)
            opt = torch.optim.SGD([fast[k] for k in upd_keys],
                                  lr=mcfg["inner_lr"])
            # extra adaptation steps (2x inner_steps) since this starts from
            # the meta-learned init with no prior global adaptation
            for _ in range(mcfg["inner_steps"] * 2):
                opt.zero_grad()
                F.cross_entropy(full_forward(sx, fast), sy).backward()
                opt.step()
            with torch.no_grad():
                pred = full_forward(qx, fast).argmax(1)
        yq, pr = qy.cpu().numpy(), pred.cpu().numpy()
        accs.append(accuracy_score(yq, pr))
        f1s.append(f1_score(yq, pr, average="macro", zero_division=0))
    return {f"{tag}_episodic_accuracy": float(np.mean(accs)),
            f"{tag}_episodic_f1_macro": float(np.mean(f1s)),
            f"{tag}_episodes": mcfg["eval_episodes"], f"{tag}_n_way": n_way}


# =============================================================================
# 5C. QUANTUM MACHINE LEARNING -- LEVEL A (QSVC, VQC, QNN, HQCNN)
# =============================================================================
# Level-A quantum extension (laptop-friendly, simulator only, no GPU):
#
#   30 SelectKBest features --(SelectKBest on TRAIN, k = q)--> top-q features
#        --(robust min-max, 1st/99th percentile)--> rotation angles in [0, pi]
#        --> q-qubit circuit (q = 4-8)
#
#   QSVC  : fidelity quantum kernel  K(x,x') = |<phi(x)|phi(x')>|^2  with a
#           ZZ-feature-map (or angle) encoding, fed to a classical SVM with a
#           precomputed kernel. No trainable circuit parameters.
#   VQC   : angle encoding + StronglyEntanglingLayers ansatz U(theta); the
#           class probabilities are read out DIRECTLY from the measured
#           computational-basis distribution (basis states assigned to classes
#           round-robin). Fully quantum model -- no classical trainable layer.
#   QNN   : data re-uploading quantum neural network (encoding repeated before
#           every variational layer) with Pauli-Z expectation readout and a
#           single linear layer mapping q expectations to class logits.
#   HQCNN : hybrid quantum-classical neural network ("dressed" quantum
#           circuit, Mari et al. 2020): classical pre-net -> PQC -> classical
#           post-net (MLP), trained end-to-end by back-propagation.
#
#   Classical controls (optional, cfg["quantum"]["classical_controls"]):
#   SVM-q (RBF-SVC) and MLP-q trained on EXACTLY the same q angle-encoded
#   features and the same training subsample as the quantum models, so any
#   quantum/classical gap is not confounded by input dimension or data size.
#
# Because circuit simulation cost grows with the number of training samples
# (QSVC kernel is O(n^2)), quantum models are trained on a class-balanced
# subsample of the (already balanced) training set; evaluation, zero-day
# detection and XAI use the same protocol as every other model.
# =============================================================================
try:
    import pennylane as qml
    HAS_PL = True
except Exception:                                     # PennyLane not installed
    qml = None
    HAS_PL = False

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def model_regime(name):
    """Family label of a model, used for colours and report columns."""
    if name in BASE_MODELS:
        return "base"
    if name in META_MODELS:
        return "meta"
    if name in QUANTUM_MODELS:
        return "quantum"
    if name in QUANTUM_B_MODELS:
        return "quantum-B"
    if name in QCONTROL_MODELS:
        return "q-control"
    return "other"


REGIME_COLORS = {"base": "#3070b3", "meta": "#d62728",
                 "quantum": "#ff7f0e", "quantum-B": "#2ca02c",
                 "q-control": "#7f7f7f", "other": "#bbbbbb"}


def stratified_subsample(y, per_class, rng):
    """Indices of at most `per_class` random rows from every class in y."""
    idx = []
    for c in np.unique(y):
        ci = np.where(y == c)[0]
        idx.append(rng.choice(ci, size=min(len(ci), per_class), replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


class QFeatureReducer:
    """Stage-3 'additional dimensionality reduction for quantum':
    SelectKBest(f_classif, k=q) over the 30 already-selected features (fit on
    TRAIN only), then a robust min-max map of each kept feature to a rotation
    angle in [0, pi] (bounds = 1st/99th train percentiles, values clipped)."""

    def __init__(self, X_tr, y_tr, feat_names, q):
        q = min(q, X_tr.shape[1])
        sel = SelectKBest(f_classif, k=q).fit(X_tr, y_tr)
        self.idx = sel.get_support(indices=True)
        self.names = [feat_names[i] for i in self.idx]
        self.scores = sel.scores_[self.idx]
        Xq = X_tr[:, self.idx]
        self.lo = np.percentile(Xq, 1, axis=0)
        self.hi = np.percentile(Xq, 99, axis=0)
        self.hi = np.where(self.hi - self.lo < 1e-9, self.lo + 1.0, self.hi)
        self.q = q

    def reduce(self, X):
        """30-feature (standardised) matrix -> q-feature matrix."""
        return np.asarray(X)[:, self.idx]

    def angles(self, Xq):
        """q-feature matrix -> rotation angles in [0, pi]."""
        z = (np.asarray(Xq, dtype=float) - self.lo) / (self.hi - self.lo)
        return np.clip(z, 0.0, 1.0) * np.pi


# ---- circuit building blocks -------------------------------------------------
def _zz_feature_map(x, wires, reps):
    """Havlicek et al. (2019) ZZ feature map (Qiskit ZZFeatureMap analogue):
    [H . RZ(2 x_i)] on every qubit, then exp(-i (pi-x_i)(pi-x_j) Z_i Z_j) on
    neighbouring pairs, repeated `reps` times. Supports batched x (B, q)."""
    for _ in range(reps):
        for i in wires:
            qml.Hadamard(wires=i)
            qml.RZ(2.0 * x[..., i], wires=i)
        for i in wires[:-1]:
            qml.IsingZZ(2.0 * (np.pi - x[..., i]) * (np.pi - x[..., i + 1]),
                        wires=[i, i + 1])


class QuantumKernel:
    """Fidelity quantum kernel computed from simulated statevectors:
    K(x, x') = |<phi(x)|phi(x')>|^2. Each sample needs ONE circuit run (its
    statevector); the Gram matrix is then a single matrix product, which is
    far cheaper than running one compute-uncompute circuit per pair and gives
    identical values on an exact simulator."""

    def __init__(self, n_qubits, qcfg):
        self.n = n_qubits
        wires = list(range(n_qubits))
        dev = qml.device(qcfg["device"], wires=n_qubits)
        fmap, reps = qcfg["feature_map"], qcfg["fmap_reps"]
        self.bw = float(qcfg.get("kernel_bandwidth", 1.0))

        @qml.qnode(dev, interface="numpy")
        def state(x):
            if fmap == "zz":
                _zz_feature_map(x, wires, reps)
            else:                                   # plain angle encoding
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
            return qml.state()
        self._state = state

    def states(self, Xa, chunk=2048):
        """(n, q) angles -> (n, 2^q) complex statevectors (batched)."""
        Xa = np.asarray(Xa, dtype=float) * self.bw     # kernel bandwidth
        out = [np.asarray(self._state(Xa[i:i + chunk])).reshape(-1, 2 ** self.n)
               for i in range(0, len(Xa), chunk)]
        return np.vstack(out)

    @staticmethod
    def gram(SA, SB):
        """Kernel matrix |<a|b>|^2 between two sets of statevectors."""
        return np.abs(SA.conj() @ SB.T) ** 2


def _build_qnode(kind, n, layers, dev_name):
    """Create the PennyLane QNode (torch interface, backprop) for VQC / QNN /
    HQCNN and return (qnode, weight_shape). All three share the same
    StronglyEntanglingLayers ansatz so that differences come from encoding /
    readout / classical parts only."""
    dev = qml.device(dev_name, wires=n)
    wires = list(range(n))
    wshape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=n)

    if kind == "VQC":
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            qml.AngleEmbedding(x, wires=wires, rotation="Y")     # U_phi(x)
            qml.StronglyEntanglingLayers(w, wires=wires)          # U(theta)
            return qml.probs(wires=wires)                         # 2^q probs
    elif kind == "QNN":
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            for l in range(layers):                               # re-uploading
                qml.AngleEmbedding(x, wires=wires, rotation="Y")
                qml.StronglyEntanglingLayers(w[l:l + 1], wires=wires)
            return [qml.expval(qml.PauliZ(i)) for i in wires]
    elif kind == "HQCNN":
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(x, w):
            for i in wires:                                       # |+>^q
                qml.Hadamard(wires=i)
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(w, wires=wires)
            return [qml.expval(qml.PauliZ(i)) for i in wires]
    else:
        raise ValueError(kind)
    return circuit, wshape


class QTorchModel(nn.Module):
    """Torch module wrapping one parameterised quantum circuit plus its
    (optional) classical layers. Split into q_forward (classical pre-net +
    circuit = raw quantum measurement statistics) and c_forward (readout ->
    class logits) so that finite-shot noise can be injected in between."""

    def __init__(self, kind, n, n_out, qcfg):
        super().__init__()
        self.kind, self.n, self.n_out = kind, n, n_out
        self.circuit, wshape = _build_qnode(kind, n, qcfg["layers"],
                                            qcfg["device"])
        self.qw = nn.Parameter(torch.rand(wshape, dtype=torch.float64)
                               * 2 * np.pi)
        if kind == "VQC":
            if 2 ** n < n_out:
                raise ValueError(f"VQC needs 2^q >= n_classes ({2**n} < {n_out})")
            M = torch.zeros(2 ** n, n_out, dtype=torch.float64)
            M[torch.arange(2 ** n), torch.arange(2 ** n) % n_out] = 1.0
            self.register_buffer("M", M)                          # state->class
        elif kind == "QNN":
            self.readout = nn.Linear(n, n_out).double()
        elif kind == "HQCNN":
            self.pre = nn.Sequential(nn.Linear(n, n), nn.Tanh()).double()
            self.post = nn.Sequential(nn.Linear(n, 32), nn.ReLU(),
                                      nn.Dropout(0.1),
                                      nn.Linear(32, n_out)).double()

    def q_forward(self, x):
        """(B, q) angles -> (B, 2^q) probs (VQC) or (B, q) <Z> expectations."""
        if self.kind == "HQCNN":
            x = self.pre(x) * (np.pi / 2)            # classical pre-processing
        out = self.circuit(x, self.qw)
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=-1)
        return out.reshape(x.shape[0], -1).to(torch.float64)

    def c_forward(self, qout):
        """Quantum statistics -> class logits."""
        if self.kind == "VQC":
            return torch.log((qout @ self.M).clamp_min(1e-10))
        if self.kind == "QNN":
            return self.readout(qout)
        return self.post(qout)

    def forward(self, x):
        return self.c_forward(self.q_forward(x))


def emulate_shots(qout, kind, shots, rng):
    """Replace exact simulator statistics with finite-shot estimates:
    multinomial sampling of basis-state probabilities (VQC), binomial sampling
    of each <Z> = 2 p(0) - 1 (QNN/HQCNN), or binomial sampling of each kernel
    entry, i.e. the all-zero probability of a compute-uncompute circuit (QSVC)."""
    if kind == "VQC":
        p = np.clip(qout, 0, None)
        p = p / p.sum(axis=1, keepdims=True)
        return rng.multinomial(shots, p) / shots
    if kind == "QSVC":
        return rng.binomial(shots, np.clip(qout, 0.0, 1.0)) / shots
    p0 = np.clip((1.0 + qout) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(shots, p0) / shots - 1.0


def train_qtorch(kind, Xa_tr, y_tr, Xa_va, y_va, n, n_out, qcfg):
    """Adam training of a VQC / QNN / HQCNN with early stopping on the
    validation cross-entropy (best weights restored)."""
    set_seed()
    model = QTorchModel(kind, n, n_out, qcfg)
    opt = torch.optim.Adam(model.parameters(), lr=qcfg["lr"])
    Xt = torch.as_tensor(Xa_tr, dtype=torch.float64)
    yt = torch.as_tensor(y_tr, dtype=torch.long)
    Xv = torch.as_tensor(Xa_va, dtype=torch.float64)
    yv = torch.as_tensor(y_va, dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)
    best, best_loss, patience = None, np.inf, 0
    history = {"train": [], "val": []}
    t0 = time.time()
    for _ in tqdm(range(qcfg["epochs"]), desc=f"\t\t\t[{kind}]", unit="epoch",
                  leave=False):
        model.train()
        perm = torch.randperm(len(Xt), generator=g)
        losses = []
        for i in range(0, len(Xt), qcfg["batch_size"]):
            b = perm[i:i + qcfg["batch_size"]]
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            vloss = F.cross_entropy(model(Xv), yv).item()
        history["train"].append(float(np.mean(losses)))
        history["val"].append(float(vloss))
        if vloss < best_loss - 1e-4:
            best_loss, best, patience = vloss, copy.deepcopy(model.state_dict()), 0
        else:
            patience += 1
            if patience >= qcfg["patience"]:
                break
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model, time.time() - t0, history


class QuantumWrapper:
    """Uniform predict_proba interface for the quantum models and their
    classical q-feature controls.

    - predict_proba(X)      : X in the 30-feature space (used by evaluation /
                              zero-day stages, like every other model)
    - predict_proba_q(Xq)   : X in the q-feature space (used by SHAP/LIME so
                              explanations are over the q features the
                              circuit actually sees)
    - predict_proba_eval(X) : like predict_proba, but with finite-shot
                              measurement noise when qcfg['eval_shots'] is set
    Output is always (n, n_space) -- local classes are scattered back into the
    full label space with zeros for classes not trained on."""

    def __init__(self, name, kind, core, reducer, classes, n_space, qcfg,
                 extra=None):
        self.name, self.kind, self.core, self.reducer = name, kind, core, reducer
        self.classes = np.asarray(classes, dtype=int)
        self.n_space, self.qcfg, self.extra = n_space, qcfg, extra or {}
        self.classes_ = np.arange(n_space)

    def _local(self, Xa, shots=None):
        rng = np.random.default_rng(SEED)
        if self.kind == "QSVC":
            qk, S_tr, svc = self.core
            Kx = QuantumKernel.gram(qk.states(Xa), S_tr)
            if shots:
                Kx = emulate_shots(Kx, "QSVC", shots, rng)
            return svc.predict_proba(Kx)
        if self.kind in ("SVM-q", "MLP-q"):
            return self.core.predict_proba(Xa)
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xa), 4096):
                xb = torch.as_tensor(Xa[i:i + 4096], dtype=torch.float64)
                qo = self.core.q_forward(xb)
                if shots:
                    qo = torch.as_tensor(
                        emulate_shots(qo.numpy(), self.kind, shots, rng))
                outs.append(F.softmax(self.core.c_forward(qo), dim=1).numpy())
        return np.vstack(outs)

    def _full(self, p):
        full = np.zeros((p.shape[0], self.n_space))
        full[:, self.classes] = p
        return sanitize_proba(full)

    def predict_proba_q(self, Xq, shots=None):
        Xa = self.reducer.angles(np.atleast_2d(Xq))
        return self._full(self._local(Xa, shots))

    def predict_proba(self, X):
        return self.predict_proba_q(self.reducer.reduce(np.atleast_2d(X)))

    def predict_proba_eval(self, X):
        return self.predict_proba_q(self.reducer.reduce(np.atleast_2d(X)),
                                    shots=self.qcfg.get("eval_shots"))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def train_quantum_suite(X_tr, y_tr, X_va, y_va, reducer, n_space, qcfg,
                        tag=""):
    """Train QSVC, VQC, QNN, HQCNN (+ SVM-q / MLP-q controls) on class-balanced
    subsamples of (X_tr, y_tr). Labels may be global ids or a remapped 0..K-1
    space (zero-day known-only re-fit); n_space is the size of that space.
    Returns {name: (QuantumWrapper, train_time_s, history_or_None)}."""
    rng = np.random.RandomState(SEED)
    n = reducer.q
    classes = np.unique(y_tr)                    # space ids actually present
    to_local = {int(c): i for i, c in enumerate(classes)}
    K = len(classes)

    Xa_tr = reducer.angles(reducer.reduce(X_tr))
    Xa_va = reducer.angles(reducer.reduce(X_va))
    yl_tr = np.array([to_local[int(c)] for c in y_tr])
    m_va = np.isin(y_va, classes)
    Xa_va, yl_va = Xa_va[m_va], np.array([to_local[int(c)] for c in y_va[m_va]])

    i_k = stratified_subsample(yl_tr, qcfg["qsvc_per_class"], rng)   # kernel set
    i_v = stratified_subsample(yl_tr, qcfg["train_per_class"], rng)  # VQC & co.
    i_val = stratified_subsample(yl_va, qcfg["val_per_class"], rng)
    log(f"    [quantum{tag}] q={n} qubits, {K} classes, "
        f"QSVC n={len(i_k)}, VQC/QNN/HQCNN n={len(i_v)}, val n={len(i_val)}")

    suite = {}
    mk = lambda name, kind, core, extra=None: QuantumWrapper(
        name, kind, core, reducer, classes, n_space, qcfg, extra)

    # ---- QSVC -------------------------------------------------------------
    t0 = time.time()
    qk = QuantumKernel(n, qcfg)
    S_tr = qk.states(Xa_tr[i_k])
    K_tr = QuantumKernel.gram(S_tr, S_tr)
    svc = SVC(kernel="precomputed", C=qcfg["qsvc_C"], probability=True,
              random_state=SEED).fit(K_tr, yl_tr[i_k])
    suite["QSVC"] = (mk("QSVC", "QSVC", (qk, S_tr, svc),
                        {"K_train": K_tr, "y_kernel": yl_tr[i_k]}),
                     time.time() - t0, None)

    # ---- VQC / QNN / HQCNN --------------------------------------------------
    for kind in ("VQC", "QNN", "HQCNN"):
        model, t, hist = train_qtorch(kind, Xa_tr[i_v], yl_tr[i_v],
                                      Xa_va[i_val], yl_va[i_val], n, K, qcfg)
        suite[kind] = (mk(kind, kind, model), t, hist)

    # ---- classical controls on the identical q-feature subsample -----------
    if qcfg.get("classical_controls", True):
        t0 = time.time()
        svm = SVC(kernel="rbf", C=qcfg["qsvc_C"], gamma="scale",
                  probability=True, random_state=SEED).fit(Xa_tr[i_k], yl_tr[i_k])
        suite["SVM-q"] = (mk("SVM-q", "SVM-q", svm), time.time() - t0, None)
        t0 = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000,
                            random_state=SEED
                            ).fit(Xa_tr[i_v], yl_tr[i_v])
        suite["MLP-q"] = (mk("MLP-q", "MLP-q", mlp), time.time() - t0, None)
    return suite


def quantum_n_params(wrapper):
    """Number of trainable parameters (circuit / total) for the report."""
    if isinstance(wrapper.core, QTorchModel):
        total = sum(p.numel() for p in wrapper.core.parameters())
        return int(wrapper.core.qw.numel()), int(total)
    return 0, 0


# ---- quantum figures / artefacts ---------------------------------------------
def save_circuit_drawings(reducer, qcfg, n_classes, path):
    """Text drawings of every circuit (for the paper's appendix)."""
    n = reducer.q
    x = np.linspace(0.3, 2.8, n)
    lines = [f"Level-A quantum circuits  (q = {n} qubits, "
             f"features: {', '.join(reducer.names)})", ""]
    try:
        qk = QuantumKernel(n, qcfg)
        lines += [f"--- QSVC feature map ({qcfg['feature_map']}, "
                  f"reps={qcfg['fmap_reps']}) ---",
                  qml.draw(qk._state, decimals=2)(x), ""]
        for kind in ("VQC", "QNN", "HQCNN"):
            m = QTorchModel(kind, n, max(2, min(n_classes, 2 ** n)), qcfg)
            lines += [f"--- {kind} ---",
                      qml.draw(m.circuit, decimals=2, max_length=160,
                               show_matrices=False)(
                          torch.as_tensor(x), m.qw.detach()), ""]
    except Exception as e:
        lines.append(f"(drawing failed: {e})")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_quantum_losses(q_histories, ds_name, path):
    """Train / validation loss curves of VQC, QNN and HQCNN."""
    q_histories = {k: v for k, v in q_histories.items() if v}
    if not q_histories:
        return
    fig, axes = plt.subplots(1, len(q_histories),
                             figsize=(4.5 * len(q_histories), 3.6),
                             squeeze=False)
    for ax, (name, h) in zip(axes[0], q_histories.items()):
        ep = np.arange(1, len(h["train"]) + 1)
        ax.plot(ep, h["train"], "-o", ms=3, label="train", color="#7b3fb5")
        ax.plot(ep, h["val"], "-s", ms=3, label="validation", color="#d62728")
        ax.set_title(name); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy")
        ax.legend(fontsize=8)
    fig.suptitle(f"{ds_name} -- quantum model learning curves")
    fig.tight_layout()
    savefig(fig, path)


def plot_quantum_inputs(reducer, X_tr, y_tr, class_names, kernel_extra,
                        ds_name, figdir):
    """(a) angle-encoded q-feature matrix (sorted by class) and (b) the QSVC
    training Gram matrix sorted by class -- block structure along the
    diagonal shows how well the quantum kernel separates the classes."""
    rng = np.random.RandomState(SEED)
    idx = stratified_subsample(y_tr, 15, rng)
    idx = idx[np.argsort(y_tr[idx], kind="stable")]
    A = reducer.angles(reducer.reduce(X_tr[idx]))
    fig, ax = plt.subplots(figsize=(1.6 + 0.7 * reducer.q, 6))
    im = ax.imshow(A, aspect="auto", cmap="Blues", vmin=0, vmax=np.pi)
    ax.set_xticks(range(reducer.q))
    ax.set_xticklabels([n[:14] for n in reducer.names], rotation=45, ha="right")
    ax.set_ylabel("Samples (sorted by class)")
    fig.colorbar(im, ax=ax, label="angle (rad)")
    ax.set_title(f"{ds_name} -- quantum input features (q={reducer.q})")
    savefig(fig, os.path.join(figdir, "quantum_feature_matrix.png"))

    if kernel_extra and "K_train" in kernel_extra:
        K, yk = kernel_extra["K_train"], kernel_extra["y_kernel"]
        o = np.argsort(yk, kind="stable")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(K[np.ix_(o, o)], cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="K(x, x')")
        bounds = np.where(np.diff(yk[o]) != 0)[0] + 0.5
        for b in bounds:
            ax.axhline(b, color="w", lw=0.5); ax.axvline(b, color="w", lw=0.5)
        ax.set_title(f"{ds_name} -- QSVC quantum kernel (train, sorted by class)")
        savefig(fig, os.path.join(figdir, "quantum_kernel_matrix.png"))


def plot_quantum_vs_classical(perf_df, zdet_df, title, path):
    """Pipeline step 8: Accuracy, Macro-F1 and zero-day AUROC (MSP) for every
    model, bars coloured by model family (base / meta / quantum / q-control)."""
    if perf_df is None or perf_df.empty:
        return
    df = perf_df.groupby("Model")[["Accuracy", "F1"]].mean()
    if zdet_df is not None and not zdet_df.empty:
        z = zdet_df[zdet_df["Score"] == "MSP"].groupby("Model")["AUROC"].mean()
        df["Zero-day AUROC"] = z
    df = df.reindex([m for m in ALL_MODELS if m in df.index])
    metrics = [c for c in ["Accuracy", "F1", "Zero-day AUROC"] if c in df.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6),
                             squeeze=False, sharey=True)
    from matplotlib.patches import Patch
    for ax, met in zip(axes[0], metrics):
        vals = df[met].values
        cols = [REGIME_COLORS[model_regime(m)] for m in df.index]
        ax.bar(np.arange(len(df)), np.nan_to_num(vals), color=cols)
        ax.set_xticks(np.arange(len(df)))
        ax.set_xticklabels(df.index, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title("Macro-F1" if met == "F1" else met)
    present = sorted({model_regime(m) for m in df.index},
                     key=list(REGIME_COLORS).index)
    labels = {"base": "classical base", "meta": "meta-learner",
              "quantum": "quantum (Level A)",
              "quantum-B": "hybrid quantum ProtoNet (Level B)",
              "q-control": "classical on q features"}
    axes[0][0].legend(handles=[Patch(color=REGIME_COLORS[r], label=labels.get(r, r))
                               for r in present], fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("Metric value")
    fig.suptitle(title)
    fig.tight_layout()
    savefig(fig, path)


# =============================================================================
# 5D. QUANTUM MACHINE LEARNING -- LEVEL B
#     HYBRID QUANTUM PROTOTYPICAL NETWORKS (few-shot / zero-day)
# =============================================================================
# Level B turns each Level-A quantum model into the EMBEDDING NETWORK of a
# prototypical network that is META-TRAINED episodically (N-way K-shot), so
# the quantum circuit itself is optimised for few-shot generalisation:
#
#   x (q features) --encoding--> |psi(x)> --PQC U(theta)--> z = f_theta(x)
#   support set S_k  ->  prototype c_k = (1/K) sum_{i in S_k} z_i
#   query x          ->  p(y=k|x) = softmax_k( -tau * ||z_x - c_k||^2 )
#
#   QSVC-PN  : TRAINABLE QUANTUM-KERNEL prototypical network. The embedding is
#              the quantum state itself (|psi(x)><psi(x)| in Hilbert space):
#              ZZ feature map with learnable per-feature bandwidths, with a
#              variational layer between the two data-encoding repetitions
#              (trainable embedding kernel, Hubregtsen et al. 2022). The
#              prototype distance is the kernel-mean distance
#              ||phi(x) - mu_k||^2 = 1 - 2 mean_i K(x,x_i) + mean_ij K(x_i,x_j)
#              -- the Level-B, meta-learned counterpart of the Level-A QSVC.
#   VQC-PN   : fully quantum embedding. Encoding + StronglyEntanglingLayers;
#              z = expectations of d Pauli observables (single-qubit X/Y/Z and
#              neighbouring ZZ/XX correlators). No classical parameters.
#   QNN-PN   : data re-uploading circuit; z = Linear(<X>,<Y>,<Z> per qubit)->d.
#   HQCNN-PN : dressed circuit: classical pre-net -> PQC -> classical
#              post-net -> z in R^d (d = 8-32).
#   MLP-PN   : CLASSICAL control -- the same prototypical network with an MLP
#              embedding on the SAME q features and the SAME episodes, isolating
#              the effect of the quantum embedding.
#
# Like the classical meta-learners, Level-B models are meta-trained on the
# known classes only (meta_pool, zero-day excluded). At test time they need
# NO gradient steps: new or withheld attack families are handled just by
# computing prototypes from a few labelled examples (few-shot), and
# open-set zero-day detection uses MSP / entropy / margin over known
# prototypes, exactly as for every other model.
# =============================================================================
QUANTUM_B_KINDS = {"QSVC-PN": "QK", "VQC-PN": "VQC", "QNN-PN": "QNN",
                   "HQCNN-PN": "HQCNN", "MLP-PN": "MLP"}


def _pauli_pool(n):
    """Observable pool for the pure-quantum (VQC-PN) embedding: X, Y, Z on
    every qubit, then neighbouring ZZ and XX correlators."""
    pool = ([qml.PauliZ(i) for i in range(n)] +
            [qml.PauliX(i) for i in range(n)] +
            [qml.PauliY(i) for i in range(n)])
    pool += [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n - 1)]
    pool += [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(n - 1)]
    return pool


def _encode(x, wires, encoding):
    """Classical-to-quantum encoding (pipeline step 3): angle (RY) or
    amplitude (q features padded to 2^q amplitudes, L2-normalised)."""
    if encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=wires, pad_with=0.0, normalize=True)
    else:
        qml.AngleEmbedding(x, wires=wires, rotation="Y")


class QProtoEmbed(nn.Module):
    """Embedding network f_theta of a Level-B prototypical network.
    kind: QK (quantum-kernel state embedding), VQC, QNN, HQCNN or MLP.
    embed(x) returns (B, 2^q) complex statevectors for QK, else (B, d) reals."""

    def __init__(self, kind, n, bcfg):
        super().__init__()
        self.kind, self.n = kind, n
        d, L = int(bcfg["embed_dim"]), int(bcfg["layers"])
        enc = bcfg.get("encoding", "angle")
        wires = list(range(n))
        # learnable inverse temperature of the distance softmax (all kinds,
        # incl. the classical control, so the comparison is like-for-like)
        self.log_tau = nn.Parameter(torch.tensor(np.log(bcfg["tau_init"]),
                                                 dtype=torch.float64))
        if kind == "MLP":
            self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(),
                                     nn.Linear(64, d)).double()
            self.d = d
            return
        dev = qml.device(bcfg["device"], wires=n)

        if kind == "QK":
            reps = int(bcfg["fmap_reps"])
            self.scale = nn.Parameter(torch.full(
                (n,), float(bcfg["kernel_bandwidth_init"]), dtype=torch.float64))
            wshape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n)
            self.qw = nn.Parameter(torch.rand((max(reps - 1, 1),) + tuple(wshape),
                                              dtype=torch.float64) * 0.2)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(x, s, w):
                for r in range(reps):
                    _zz_feature_map(x * s, wires, 1)          # data layer
                    if r < reps - 1:                          # trainable layer
                        qml.StronglyEntanglingLayers(w[r], wires=wires)
                return qml.state()
            self.circuit, self.d = circuit, 2 ** n
            return

        wshape = qml.StronglyEntanglingLayers.shape(n_layers=L, n_wires=n)
        self.qw = nn.Parameter(torch.rand(wshape, dtype=torch.float64) * 2 * np.pi)
        if kind == "VQC":
            pool = _pauli_pool(n)
            if d > len(pool):
                log(f"    [Level B] VQC-PN: embed_dim {d} > {len(pool)} "
                    f"observables available with q={n}; using {len(pool)}")
            obs = pool[:d]
            self.d = len(obs)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(x, w):
                _encode(x, wires, enc)
                qml.StronglyEntanglingLayers(w, wires=wires)
                return [qml.expval(o) for o in obs]
        else:
            obs = ([qml.PauliX(i) for i in wires] + [qml.PauliY(i) for i in wires]
                   + [qml.PauliZ(i) for i in wires])
            if kind == "QNN":
                # re-uploading needs a re-applicable encoding -> always angle
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def circuit(x, w):
                    for l in range(L):
                        qml.AngleEmbedding(x, wires=wires, rotation="Y")
                        qml.StronglyEntanglingLayers(w[l:l + 1], wires=wires)
                    return [qml.expval(o) for o in obs]
                self.post = nn.Linear(3 * n, d).double()
            else:                                             # HQCNN
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def circuit(x, w):
                    if enc != "amplitude":
                        for i in wires:
                            qml.Hadamard(wires=i)
                    _encode(x, wires, enc)
                    qml.StronglyEntanglingLayers(w, wires=wires)
                    return [qml.expval(o) for o in obs]
                self.pre = nn.Sequential(nn.Linear(n, n), nn.Tanh()).double()
                self.post = nn.Sequential(nn.Linear(3 * n, 32), nn.ReLU(),
                                          nn.Linear(32, d)).double()
            self.d = d
        self.circuit = circuit

    # -- raw quantum statistics (where finite-shot noise is injected) -------
    def _quantum(self, x):
        if self.kind == "QK":
            return self.circuit(x, self.scale, self.qw)
        if self.kind == "HQCNN":
            x = self.pre(x) * (np.pi / 2) + np.pi / 2       # back into [0, pi]
        out = self.circuit(x, self.qw)
        return torch.stack(out, dim=-1).reshape(x.shape[0], -1).to(torch.float64)

    def embed(self, x, shots=None, rng=None):
        """(B, q) angles -> embedding. With `shots`, expectation values are
        replaced by binomial finite-shot estimates (QK: handled in kernel)."""
        if self.kind == "MLP":
            return self.net(x)
        qo = self._quantum(x)
        if self.kind == "QK":
            return qo
        if shots:
            qo = torch.as_tensor(emulate_shots(qo.detach().numpy(), "expval",
                                               shots, rng))
        return self.post(qo) if self.kind in ("QNN", "HQCNN") else qo


def _kernel(SA, SB):
    """Fidelity kernel |<a|b>|^2 between two batches of statevectors (torch)."""
    return (SA.conj() @ SB.T).abs() ** 2


def proto_distances(net, z_q, z_s, ys, n_cls, shots=None, rng=None):
    """Squared distance of every query to every class prototype.
    QK: kernel-mean distance in Hilbert space; others: squared Euclidean
    distance to the mean embedding c_k."""
    if net.kind == "QK":
        Kqs, Kss = _kernel(z_q, z_s), _kernel(z_s, z_s)
        if shots:                                     # finite-shot kernel entries
            Kqs = torch.as_tensor(emulate_shots(Kqs.numpy(), "QSVC", shots, rng))
        d = []
        for k in range(n_cls):
            m = ys == k
            d.append(1.0 - 2.0 * Kqs[:, m].mean(1) + Kss[m][:, m].mean())
        return torch.stack(d, dim=1)
    protos = torch.stack([z_s[ys == k].mean(0) for k in range(n_cls)])
    return torch.cdist(z_q, protos) ** 2


def sample_episode_idx(y, pool, n_way, k_shot, q_query, rng):
    """Numpy N-way K-shot episode: support/query row indices + LOCAL labels."""
    classes = rng.choice(pool, size=min(n_way, len(pool)), replace=False)
    si, sl, qi, ql = [], [], [], []
    for j, c in enumerate(classes):
        idx = np.where(y == c)[0]
        take = rng.choice(idx, size=k_shot + q_query,
                          replace=len(idx) < k_shot + q_query)
        si.append(take[:k_shot]); sl += [j] * k_shot
        qi.append(take[k_shot:]); ql += [j] * q_query
    return (np.concatenate(si), np.array(sl), np.concatenate(qi),
            np.array(ql), classes)


def meta_train_qproto(name, Xa_tr, y_tr, class_pool, n, bcfg):
    """Episodic prototypical meta-training of one Level-B model."""
    set_seed()
    rng = np.random.RandomState(SEED)
    net = QProtoEmbed(QUANTUM_B_KINDS[name], n, bcfg)
    opt = torch.optim.Adam(net.parameters(), lr=bcfg["lr"])
    n_way = min(bcfg["n_way"], len(class_pool))
    X = torch.as_tensor(Xa_tr, dtype=torch.float64)
    history = []
    t0 = time.time()
    for _ in tqdm(range(bcfg["episodes"]), desc=f"\t\t\t[{name}]",
                  unit="episode", leave=False):
        opt.zero_grad()
        losses = []
        for _ in range(bcfg["meta_batch"]):
            si, sl, qi, ql, _ = sample_episode_idx(
                y_tr, class_pool, n_way, bcfg["k_shot"], bcfg["q_query"], rng)
            z = net.embed(X[np.concatenate([si, qi])])
            zs, zq = z[:len(si)], z[len(si):]
            dist = proto_distances(net, zq, zs, torch.as_tensor(sl), n_way)
            loss = F.cross_entropy(-net.log_tau.exp() * dist,
                                   torch.as_tensor(ql))
            (loss / bcfg["meta_batch"]).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        history.append(float(np.mean(losses)))
    net.eval()
    return net, time.time() - t0, history


class QProtoWrapper:
    """Prototype classifier built from a meta-trained Level-B embedding and a
    labelled support set (adapt_shot rows per class). Same interface as
    QuantumWrapper: predict_proba (30 features), predict_proba_q (q features,
    used by SHAP/LIME) and predict_proba_eval (finite shots if configured).
    No gradient steps are taken here -- only prototypes are computed."""

    def __init__(self, name, net, reducer, Xs, ys, n_space, bcfg):
        self.name, self.net, self.reducer = name, net, reducer
        self.n_space, self.bcfg = n_space, bcfg
        self.classes = np.unique(ys)
        self.classes_ = np.arange(n_space)
        loc = np.searchsorted(self.classes, ys)
        Xa = reducer.angles(reducer.reduce(Xs))
        with torch.no_grad():
            self.zs = net.embed(torch.as_tensor(Xa, dtype=torch.float64))
        self.ys = torch.as_tensor(loc)

    def _local(self, Xa, shots=None):
        rng = np.random.default_rng(SEED)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xa), 2048):
                xb = torch.as_tensor(Xa[i:i + 2048], dtype=torch.float64)
                zq = self.net.embed(xb, shots=shots, rng=rng)
                d = proto_distances(self.net, zq, self.zs, self.ys,
                                    len(self.classes), shots=shots, rng=rng)
                out.append(F.softmax(-self.net.log_tau.exp() * d, 1).numpy())
        return np.vstack(out)

    def predict_proba_q(self, Xq, shots=None):
        p = self._local(self.reducer.angles(np.atleast_2d(Xq)), shots)
        full = np.zeros((p.shape[0], self.n_space))
        full[:, self.classes] = p
        return sanitize_proba(full)

    def predict_proba(self, X):
        return self.predict_proba_q(self.reducer.reduce(np.atleast_2d(X)))

    def predict_proba_eval(self, X):
        return self.predict_proba_q(self.reducer.reduce(np.atleast_2d(X)),
                                    shots=self.bcfg.get("eval_shots"))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def support_set(X, y, per_class, seed=SEED):
    """adapt_shot labelled rows per class (the few-shot support set)."""
    idx = stratified_subsample(y, per_class, np.random.RandomState(seed))
    return X[idx], y[idx]


def qproto_episodic_eval(net, Xa_te, y_te, pool, bcfg, tag):
    """Episodic N-way K-shot test evaluation (same protocol as the classical
    meta-learners): prototypes from K support shots of TEST classes, scored on
    the episode's queries. Returns mean and std over episodes."""
    rng = np.random.RandomState(SEED + 7)
    pool = [c for c in pool if np.sum(y_te == c) >= bcfg["k_shot"] + 2]
    if len(pool) < 2:
        return {}
    n_way = min(bcfg["n_way"], len(pool))
    X = torch.as_tensor(Xa_te, dtype=torch.float64)
    accs, f1s = [], []
    with torch.no_grad():
        for _ in range(bcfg["eval_episodes"]):
            si, sl, qi, ql, _ = sample_episode_idx(
                y_te, np.array(pool), n_way, bcfg["k_shot"], bcfg["q_query"], rng)
            z = net.embed(X[np.concatenate([si, qi])])
            d = proto_distances(net, z[len(si):], z[:len(si)],
                                torch.as_tensor(sl), n_way)
            pr = d.argmin(1).numpy()
            accs.append(accuracy_score(ql, pr))
            f1s.append(f1_score(ql, pr, average="macro", zero_division=0))
    return {f"{tag}_episodic_accuracy": float(np.mean(accs)),
            f"{tag}_episodic_accuracy_std": float(np.std(accs)),
            f"{tag}_episodic_f1_macro": float(np.mean(f1s)),
            f"{tag}_episodic_f1_macro_std": float(np.std(f1s)),
            f"{tag}_episodes": bcfg["eval_episodes"], f"{tag}_n_way": n_way}


def qproto_n_params(net):
    """(quantum-circuit parameters, total trainable parameters)."""
    total = sum(p.numel() for p in net.parameters())
    circ = 0
    if hasattr(net, "qw"):
        circ += net.qw.numel()
    if hasattr(net, "scale"):
        circ += net.scale.numel()
    return int(circ), int(total)


# ---- Level-B figures / artefacts ----------------------------------------------
def save_levelB_circuits(nets, reducer, path):
    """Text drawings of the Level-B embedding circuits."""
    x = torch.as_tensor(np.linspace(0.3, 2.8, reducer.q))
    lines = [f"Level-B quantum embedding circuits (q = {reducer.q}, "
             f"features: {', '.join(reducer.names)})", ""]
    for name, net in nets.items():
        if net.kind == "MLP":
            continue
        try:
            args = ((x, net.scale.detach(), net.qw.detach()) if net.kind == "QK"
                    else (x, net.qw.detach()))
            lines += [f"--- {name} ---",
                      qml.draw(net.circuit, decimals=2, max_length=160,
                               show_matrices=False)(*args), ""]
        except Exception as e:
            lines.append(f"--- {name}: drawing failed ({e}) ---")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_levelB_losses(histories, ds_name, path):
    """Smoothed episodic meta-training loss of every Level-B model."""
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, h in histories.items():
        s = _smooth(h, frac=0.05)
        ax.plot(np.arange(len(s)), s, lw=1.5, label=name,
                color=None if name != "MLP-PN" else "#555555",
                ls="--" if name == "MLP-PN" else "-")
    ax.set_xlabel("Meta-training episode")
    ax.set_ylabel("Prototypical query loss (smoothed)")
    ax.set_title(f"{ds_name} -- Level-B hybrid quantum ProtoNet training")
    ax.legend(fontsize=8)
    savefig(fig, path)


def plot_levelB_embeddings(wrappers, X_te, y_te, class_names, zero_day,
                           ds_name, figdir, n_per_class=60):
    """2-D PCA of test embeddings + class prototypes for each Level-B model;
    withheld zero-day families are drawn as crosses (they were never seen in
    meta-training -- whether they form their own cluster is the point)."""
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(SEED)
    idx = stratified_subsample(y_te, n_per_class, rng)
    Xs, ys = X_te[idx], y_te[idx]
    for name, w in wrappers.items():
        try:
            Xa = w.reducer.angles(w.reducer.reduce(Xs))
            with torch.no_grad():
                z = w.net.embed(torch.as_tensor(Xa, dtype=torch.float64))
                zp = w.zs
            if w.net.kind == "QK":       # complex states -> real feature vector
                to_r = lambda s: torch.cat([s.real, s.imag], 1)
                z, zp = to_r(z), to_r(zp)
            z, zp = z.numpy(), zp.numpy()
            protos = np.stack([zp[w.ys.numpy() == k].mean(0)
                               for k in range(len(w.classes))])
            pca = PCA(2, random_state=SEED).fit(np.vstack([z, protos]))
            Z, P = pca.transform(z), pca.transform(protos)
            fig, ax = plt.subplots(figsize=(7, 6))
            cmap = plt.get_cmap("tab20")
            for j, c in enumerate(np.unique(ys)):
                m = ys == c
                ax.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6,
                           marker="x" if c in zero_day else "o",
                           color=cmap(j % 20),
                           label=class_names[c][:16] +
                           (" (zero-day)" if c in zero_day else ""))
            ax.scatter(P[:, 0], P[:, 1], s=160, marker="*", c="k",
                       edgecolors="w", label="prototypes")
            ax.set_title(f"{ds_name} -- {name} embedding space (PCA)")
            ax.legend(fontsize=6, ncol=2, loc="best")
            savefig(fig, os.path.join(figdir, f"levelB_embedding_{name}.png"))
        except Exception as e:
            log(f"    embedding plot failed for {name}: {e}")


# ---- bootstrap confidence intervals (error bars of pipeline step 8) -----------
def bootstrap_ci(fn, *arrays, B=200, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI of fn(*arrays) over resampled rows."""
    if not B:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        try:
            vals.append(fn(*[a[i] for a in arrays]))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


FAMILIES = OrderedDict([
    ("Classical", ["LightGBM", "CNN"]),
    ("Classical meta-learning", ["Reptile", "MAML", "FOMAML", "Meta-SGD",
                                 "ANIL", "BOIL", "ProtoNet"]),
    ("Level A (QSVC/VQC/QNN/HQCNN)", ["QSVC", "VQC", "QNN", "HQCNN"]),
    ("Level B (hybrid quantum ProtoNet)", ["QSVC-PN", "VQC-PN", "QNN-PN",
                                           "HQCNN-PN"]),
])
FAMILY_COLORS = ["#3070b3", "#d62728", "#ff7f0e", "#2ca02c"]


def plot_level_comparison(perf, zdet, title, path):
    """Pipeline step 8: Classical vs Level A vs Level B. For each metric the
    best model of each family is shown. One dataset -> error bars are 95%
    bootstrap CIs; several datasets -> mean +- std over datasets of each
    family's best model. The winning model's name is printed on each bar."""
    if perf is None or perf.empty:
        return
    df = perf.copy()
    if zdet is not None and not zdet.empty:
        z = zdet[zdet["Score"] == "MSP"][["Dataset", "Model", "AUROC",
                                          "AUROC_lo", "AUROC_hi"]]
        df = df.merge(z, on=["Dataset", "Model"], how="left")
    metrics = [("Accuracy", "Accuracy"), ("F1", "Macro-F1"),
               ("AUROC", "Zero-day AUROC")]
    metrics = [(m, l) for m, l in metrics if m in df.columns]
    datasets = list(dict.fromkeys(df["Dataset"]))
    fams = [(f, [m for m in ms if m in set(df["Model"])])
            for f, ms in FAMILIES.items()]
    fams = [(f, ms) for f, ms in fams if ms]
    fig, ax = plt.subplots(figsize=(3.2 + 2.6 * len(metrics), 5))
    width = 0.8 / max(len(fams), 1)
    for j, (fam, ms) in enumerate(fams):
        vals, lo, hi, labels = [], [], [], []
        for met, _ in metrics:
            sub = df[df["Model"].isin(ms)]
            best = sub.loc[sub.groupby("Dataset")[met].idxmax().dropna()]
            if best.empty:
                vals.append(np.nan); lo.append(0); hi.append(0); labels.append("")
                continue
            v = best[met].mean()
            if len(datasets) == 1 and f"{met}_lo" in best.columns:
                el = v - best[f"{met}_lo"].iloc[0]
                eh = best[f"{met}_hi"].iloc[0] - v
            else:
                el = eh = best[met].std(ddof=0) if len(best) > 1 else 0.0
            vals.append(v); lo.append(np.nan_to_num(el)); hi.append(np.nan_to_num(eh))
            labels.append(best["Model"].mode().iloc[0])
        x = np.arange(len(metrics)) + (j - (len(fams) - 1) / 2) * width
        ax.bar(x, np.nan_to_num(vals), width, yerr=[lo, hi], capsize=3,
               color=FAMILY_COLORS[j % len(FAMILY_COLORS)], label=fam)
        for xi, v, lab in zip(x, vals, labels):
            if lab and np.isfinite(v):
                ax.text(xi, 0.02, lab, rotation=90, ha="center", va="bottom",
                        fontsize=7, color="white")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Metric value (best model per family)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, frameon=False)
    savefig(fig, path)


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


def _smooth(v, frac=0.02):
    """Moving average with a window proportional to the sequence length."""
    v = np.asarray(v, dtype=float)
    w = max(1, int(len(v) * frac))
    if w <= 1 or len(v) < 3:
        return v
    return np.convolve(v, np.ones(w) / w, mode="valid")


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


# =============================================================================
# 6C. UNIFIED ZERO-DAY (OPEN-SET / NOVELTY) DETECTION FOR ALL MODELS
# =============================================================================
# Motivation
# ----------
# A "zero-day" attack is one whose ATTACK TYPE was never present in the
# training data. Detecting it is an OPEN-SET / novelty-detection problem:
# the model must flag "this looks like nothing I was trained on" WITHOUT
# being handed any labelled example of the new class. This is fundamentally
# different from the (closed-set) few-shot evaluation earlier in the file,
# where the held-out classes are still provided as a labelled support set.
#
# To make the comparison fair across EVERY model -- the two base learners
# (LightGBM, CNN) and the seven meta-learners -- this stage:
#   1. Re-fits / re-adapts each model on the KNOWN classes only, so the
#      zero-day classes are completely removed from its training signal.
#      * Base learners are re-fit from scratch on the known-only split.
#      * Meta-learners already meta-trained on `meta_pool` (= known classes),
#        so they only need a cheap known-only global adaptation; their
#        full-space predict_proba is then sliced to the known columns.
#   2. Has every model expose predict_proba over exactly the K known classes.
#   3. Derives a model-agnostic NOVELTY SCORE per test sample from that
#      probability vector (higher => more likely an unseen/zero-day attack):
#        - MSP     : 1 - max_c p(c|x)                (Hendrycks & Gimpel 2017)
#        - Entropy : normalised Shannon entropy of p(.|x)
#        - Margin  : 1 - (p_top1 - p_top2)
#   4. Treats zero-day test samples as the positive class and known-class
#      test samples as negatives, and reports the standard open-set metrics:
#      AUROC, AUPR, FPR@95%TPR, detection rate (TPR) at fixed FPR budgets,
#      and the best-F1 operating point. Because the scores are pure functions
#      of predict_proba, base and meta learners are compared apples-to-apples.
# =============================================================================
def novelty_scores(proba, methods):
    """Convert a (n, K) known-class probability matrix into per-sample novelty
    scores in [0, 1] (higher => more likely an unseen 'zero-day' class).

    Every score depends only on predict_proba, so the exact same code applies
    to LightGBM, the CNN and all seven meta-learners."""
    p = np.clip(np.asarray(proba, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)          # defensive renormalisation
    k = p.shape[1]
    out = {}
    if "msp" in methods:                          # low max-confidence => novel
        out["MSP"] = 1.0 - p.max(axis=1)
    if "entropy" in methods:                      # high uncertainty => novel
        H = -(p * np.log(p)).sum(axis=1)
        out["Entropy"] = H / np.log(k) if k > 1 else np.zeros(len(p))
    if "margin" in methods:                       # small top1-top2 gap => novel
        srt = np.sort(p, axis=1)[:, ::-1]
        gap = srt[:, 0] - (srt[:, 1] if k > 1 else 0.0)
        out["Margin"] = 1.0 - gap
    return out


def openset_detection_metrics(score, is_zeroday, target_fprs):
    """Binary novelty-detection metrics (zero-day = positive, known = negative).

    Returns a metrics dict -- AUROC, AUPR, FPR@95%TPR, detection rate (TPR)
    at each requested false-positive budget, and the best-F1 operating point
    with its threshold -- plus the ROC (fpr, tpr) arrays for curve overlays.
    Returns NaNs (and None for the ROC) when only one class is present."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 roc_curve, precision_recall_curve)
    score = np.asarray(score, dtype=float)
    is_zeroday = np.asarray(is_zeroday, dtype=int)
    res = {"AUROC": np.nan, "AUPR": np.nan, "FPR@95TPR": np.nan,
           "BestF1": np.nan, "BestF1_Precision": np.nan,
           "BestF1_Recall": np.nan, "BestF1_Threshold": np.nan}
    for tf in target_fprs:                         # pre-seed TPR@FPR columns
        res[f"TPR@FPR{int(round(tf * 100))}"] = np.nan
    if len(np.unique(is_zeroday)) < 2:             # need both pos and neg
        return res, None

    res["AUROC"] = float(roc_auc_score(is_zeroday, score))
    res["AUPR"] = float(average_precision_score(is_zeroday, score))

    fpr, tpr, _ = roc_curve(is_zeroday, score)
    # FPR at the first threshold reaching >= 95% TPR (a standard OOD metric,
    # lower is better: how many known samples get flagged to catch 95% of
    # zero-days).
    idx = int(np.searchsorted(tpr, 0.95, side="left"))
    res["FPR@95TPR"] = float(fpr[min(idx, len(fpr) - 1)])
    # detection rate (TPR) attainable while staying under each FPR budget
    for tf in target_fprs:
        j = int(np.searchsorted(fpr, tf, side="right")) - 1
        res[f"TPR@FPR{int(round(tf * 100))}"] = float(tpr[max(j, 0)])

    # best-F1 operating point over the precision-recall sweep
    prec, rec, thr = precision_recall_curve(is_zeroday, score)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    b = int(np.nanargmax(f1))
    res["BestF1"] = float(f1[b])
    res["BestF1_Precision"] = float(prec[b])
    res["BestF1_Recall"] = float(rec[b])
    # precision_recall_curve returns one fewer threshold than points
    res["BestF1_Threshold"] = (float(thr[min(b, len(thr) - 1)])
                               if len(thr) else np.nan)
    return res, (fpr, tpr)


def _slice_renorm(proba, cols):
    """Keep only the `cols` (known-class) columns of a proba matrix and
    renormalise each row to sum to 1 -- used to turn a meta-learner's
    full-label-space output into a known-only distribution."""
    p = np.asarray(proba, dtype=float)[:, cols]
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


def build_known_only_models(meta_state, X_tr, y_tr, X_va, y_va,
                            known_classes, n_classes, cfg, q_reducer=None,
                            qb_nets=None):
    """Return {model_name: pp} where each pp(X) -> (n, K) probabilities over
    ONLY the K known classes (the zero-day classes are removed from the
    training signal so the open-set test is fair).

    - Base learners (LightGBM, CNN) are RE-FIT from scratch on the known-only
      training split, with labels remapped to a contiguous 0..K-1 range.
    - Meta-learners reuse their already-meta-trained parameters (meta-training
      only ever saw the known `meta_pool` classes) and are cheaply RE-ADAPTED
      with a known-only global support set; the resulting full-space
      predict_proba is then sliced back down to the known columns."""
    known_classes = list(map(int, known_classes))
    remap = {c: i for i, c in enumerate(known_classes)}   # global id -> 0..K-1
    K = len(known_classes)

    # ---- known-only train / val splits --------------------------------------
    m_tr = np.isin(y_tr, known_classes)
    m_va = np.isin(y_va, known_classes)
    # remapped labels (0..K-1) for the base learners that train a K-way head
    Xk_tr, yk_tr = X_tr[m_tr], np.array([remap[int(c)] for c in y_tr[m_tr]])
    Xk_va, yk_va = X_va[m_va], np.array([remap[int(c)] for c in y_va[m_va]])

    known_pp = {}

    # ---- base learners: re-fit on the known classes only --------------------
    if cfg["zeroday_detect"]["retrain_base_known_only"]:
        log("    [zero-day] re-fitting known-only LightGBM ...")
        lgbm_k, _ = train_lightgbm(Xk_tr, yk_tr, Xk_va, yk_va)
        known_pp["LightGBM"] = make_full_proba(lgbm_k, K)      # (n, K)
        print("\t")

        log("    [zero-day] re-fitting known-only CNN ...")
        cnn_k, _, _ = train_cnn(Xk_tr, yk_tr, Xk_va, yk_va, K, cfg["cnn"])
        known_pp["CNN"] = TorchWrapper(cnn_k, K).predict_proba  # (n, K)
        print("\t")

    # ---- quantum models (+ q-feature controls): re-fit on known classes -----
    # Same q-feature reducer as in stage 5C; labels remapped to 0..K-1, so
    # each wrapper outputs (n, K) probabilities over the known classes only.
    if (q_reducer is not None and cfg["quantum"].get("enabled", True) and
            cfg["zeroday_detect"].get("retrain_quantum_known_only", True)):
        log("    [zero-day] re-fitting known-only quantum models ...")
        qsuite = train_quantum_suite(Xk_tr, yk_tr, Xk_va, yk_va, q_reducer,
                                     K, cfg["quantum"], tag="/known-only")
        for qname, (qw, _, _) in qsuite.items():
            known_pp[qname] = qw.predict_proba                  # (n, K)

    # ---- Level-B ProtoNets: already meta-trained on known classes only ------
    # No retraining: prototypes are simply recomputed from a known-only
    # support set (labels remapped to 0..K-1) -> (n, K) probabilities.
    if q_reducer is not None and qb_nets:
        Xs_k, ys_k = support_set(Xk_tr, yk_tr, cfg["quantum_b"]["adapt_shot"])
        for bname, net in qb_nets.items():
            known_pp[bname] = QProtoWrapper(bname, net, q_reducer, Xs_k, ys_k,
                                            K, cfg["quantum_b"]).predict_proba

    # ---- meta-learners: known-only global adaptation, then column-slice -----
    mcfg = cfg["meta"]
    cols = np.array(known_classes)                 # columns to keep (global ids)
    Xk_tr_orig, yk_tr_orig = X_tr[m_tr], y_tr[m_tr]   # ORIGINAL ids (full head)
    for algo, (params, lr_dict) in meta_state.items():
        # adapt the meta-init to the known label space using a small support set
        wrap = global_adapt(algo, params, lr_dict, Xk_tr_orig, yk_tr_orig,
                            n_classes, mcfg)
        # bind `wrap` now (default-arg trick) to avoid the classic late-binding
        # closure bug inside this loop, then slice to the K known columns
        known_pp[algo] = (lambda w, c=cols:
                          (lambda X: _slice_renorm(w.predict_proba(X), c)))(wrap)
    return known_pp


def run_zeroday_detection(ds_name, meta_state, X_tr, y_tr, X_va, y_va,
                          X_te, y_te, meta_pool, zero_day, n_classes,
                          class_names, cfg, figdir, outdir, q_reducer=None,
                          qb_nets=None):
    """Stage 6C driver. Builds known-only models, assembles an open-set test
    set (known negatives + zero-day positives), scores every model with every
    novelty score, computes the open-set detection metrics, draws the figures,
    and returns a list of metric rows (one per model x score method)."""
    zdc = cfg["zeroday_detect"]
    if not zdc.get("enabled", True) or not zero_day:
        return []                                   # feature disabled / no held-out

    known_classes = sorted(int(c) for c in meta_pool)
    if len(known_classes) < 2:                      # need a real known label space
        return []

    # ---- assemble the open-set test set: known (neg) + zero-day (pos) -------
    m_known = np.isin(y_te, known_classes)
    m_zero = np.isin(y_te, list(map(int, zero_day)))
    if m_zero.sum() == 0 or m_known.sum() == 0:
        log(f"[{ds_name}] [zero-day] not enough test samples -- skipped")
        return []

    # optionally subsample the (usually large) known negatives for speed
    rng = np.random.RandomState(SEED)
    known_idx = np.where(m_known)[0]
    cap = zdc.get("max_test_known")
    if cap and len(known_idx) > cap:
        known_idx = rng.choice(known_idx, cap, replace=False)
    zero_idx = np.where(m_zero)[0]
    sel = np.concatenate([known_idx, zero_idx])
    X_os = X_te[sel]                                # open-set evaluation inputs
    is_zero = np.concatenate([np.zeros(len(known_idx), int),   # 0 = known
                              np.ones(len(zero_idx), int)])     # 1 = zero-day
    log(f"[{ds_name}] [zero-day] open-set test: "
        f"{len(known_idx)} known vs {len(zero_idx)} zero-day samples")

    # ---- known-only models (each pp(X) -> (n, K) over known classes) --------
    known_pp = build_known_only_models(meta_state, X_tr, y_tr, X_va, y_va,
                                       known_classes, n_classes, cfg,
                                       q_reducer=q_reducer, qb_nets=qb_nets)

    methods = zdc["score_methods"]
    target_fprs = zdc["target_fprs"]
    primary = "MSP"                    # score used for the ROC-overlay figure
    rows, roc_curves, score_hist = [], {}, {}
    for name, pp in known_pp.items():
        proba = pp(X_os)                                    # (n, K)
        scores = novelty_scores(proba, methods)             # dict score->array
        for sname, sc in scores.items():
            m, roc = openset_detection_metrics(sc, is_zero, target_fprs)
            if sname == "MSP":                 # 95% bootstrap CI of AUROC
                m["AUROC_lo"], m["AUROC_hi"] = bootstrap_ci(
                    roc_auc_score, is_zero, sc, B=cfg.get("bootstrap", 0))
            rows.append({"Dataset": ds_name, "Model": name, "Score": sname,
                         "Regime": model_regime(name),
                         "N_known": int(len(known_idx)),
                         "N_zeroday": int(len(zero_idx)), **m})
            if sname == primary and roc is not None:        # cache for plots
                roc_curves[name] = roc
                score_hist[name] = (sc[is_zero == 0], sc[is_zero == 1])
        auroc = next((r["AUROC"] for r in rows
                      if r["Model"] == name and r["Score"] == primary), np.nan)
        log(f"[{ds_name}] [zero-day] {name:9s} AUROC({primary})={auroc:.3f}")

    # ---- figures ------------------------------------------------------------
    det_df = pd.DataFrame(rows)
    plot_zeroday_detection(det_df, roc_curves, score_hist, primary,
                           ds_name, figdir)
    plot_openset_metric_suite(det_df, primary, ds_name, figdir)  # full metric suite
    return rows


def plot_zeroday_detection(det_df, roc_curves, score_hist, primary,
                           ds_name, figdir):
    """Three stage-6C figures: (a) AUROC bars (models x score methods),
    (b) ROC-curve overlay for the primary score, and (c) novelty-score
    distributions (known vs zero-day) as per-model small multiples."""
    if det_df is None or det_df.empty:
        return

    # (a) AUROC bar chart, grouped by novelty score ---------------------------
    piv = det_df.pivot_table(index="Model", columns="Score", values="AUROC")
    piv = piv.reindex([m for m in ALL_MODELS if m in piv.index])
    fig, ax = plt.subplots(figsize=(10, 5))
    piv.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")     # chance level
    ax.set_ylim(0, 1.02); ax.set_ylabel("Zero-day detection AUROC")
    ax.set_title(f"{ds_name} -- zero-day (open-set) detection AUROC per model")
    ax.legend(fontsize=8, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    savefig(fig, os.path.join(figdir, "zeroday_detection_auroc.png"))

    # (b) ROC overlay for the primary novelty score ---------------------------
    if roc_curves:
        fig, ax = plt.subplots(figsize=(7, 6))
        for name in [m for m in ALL_MODELS if m in roc_curves]:
            fpr, tpr = roc_curves[name]
            ax.plot(fpr, tpr, lw=1.3,label=f"{name[:18]} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)   # chance diagonal
        ax.set_xlabel("False positive rate (known flagged as zero-day)")
        ax.set_ylabel("True positive rate (zero-day detected)")
        ax.set_title(f"{ds_name} -- zero-day detection ROC ({primary})")
        ax.legend(fontsize=7, loc="lower right")
        savefig(fig, os.path.join(figdir, "zeroday_detection_roc.png"))

    # (c) novelty-score distributions per model (known vs zero-day) -----------
    if score_hist:
        names = [m for m in ALL_MODELS if m in score_hist]
        ncol = 3
        nrow = int(np.ceil(len(names) / ncol))
        bins = np.linspace(0, 1, 30)
        # 1. Combined figure containing all subfigures
        fig, axes = plt.subplots(nrow,ncol,figsize=(4 * ncol, 3 * nrow),squeeze=False)
        fig.subplots_adjust(hspace=0.5)
        for k, name in enumerate(names):
            # Position of the subplot
            ax = axes[k // ncol][k % ncol]
            kn, zd = score_hist[name]
            # Plot known samples
            ax.hist(kn,bins=bins,alpha=0.6,density=True,color="#3070b3",label="known")
            # Plot zero-day samples
            ax.hist(zd,bins=bins,alpha=0.6,density=True,color="#d62728",label="zero-day")
            ax.set_title(name, fontsize=9)
            ax.set_xlabel(f"{primary} score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            # 2. Save each subfigure separately
            fig_single, ax_single = plt.subplots(figsize=(6, 4))
            ax_single.hist(kn,bins=bins,alpha=0.6,density=True,color="#3070b3",label="known")
            ax_single.hist(zd,bins=bins,alpha=0.6,density=True,color="#d62728",label="zero-day")
            ax_single.set_title(f"{ds_name} -- {name}",fontsize=11)
            ax_single.set_xlabel(f"{primary} score")
            ax_single.set_ylabel("Density")
            ax_single.legend(fontsize=8)
            fig_single.tight_layout()
            savefig(fig_single,os.path.join(figdir,f"zeroday_score_distribution_{name}.png"))
            plt.close(fig_single)

        # Hide unused subplots
        for k in range(len(names), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")

        fig.suptitle(f"{ds_name} -- novelty-score distributions ({primary})")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        # Save the combined figure
        savefig(fig, os.path.join(figdir, "zeroday_score_distributions.png"))    


def plot_openset_metric_suite(det_df, primary, ds_name, figdir):
    """Publication figures covering the FULL stage-6C open-set metric suite
    (the plots above only show AUROC). Produces:

      (a) a higher-is-better heatmap of AUROC / AUPR / detection-rate@FPR /
          best-F1 per model, for the primary novelty score;
      (b) an FPR@95%TPR bar chart (lower is better -- the fraction of known
          traffic falsely flagged in order to catch 95% of the zero-days),
          coloured by base vs meta regime;
      (c) a per-score AUROC/AUPR comparison so the three novelty scores can be
          judged side by side."""
    if det_df is None or det_df.empty:
        return
    order = [m for m in ALL_MODELS if m in set(det_df["Model"])]

    # (a) higher-is-better metric heatmap (primary score) ---------------------
    sub = det_df[det_df["Score"] == primary].set_index("Model").reindex(order)
    hib = [c for c in ["AUROC", "AUPR", "TPR@FPR5", "TPR@FPR10", "BestF1"]
           if c in sub.columns]
    if hib:
        heatmap_df(
            sub[hib],
            f"{ds_name} -- zero-day open-set metrics ({primary}; higher=better)",
            os.path.join(figdir, "zeroday_metric_suite_heatmap.png"),
            fmt=".3f", cmap="viridis")

    # (b) FPR@95%TPR bar chart (lower is better) ------------------------------
    if "FPR@95TPR" in sub.columns and sub["FPR@95TPR"].notna().any():
        from matplotlib.patches import Patch
        colors = [REGIME_COLORS[model_regime(m)] for m in sub.index]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(np.arange(len(sub)), sub["FPR@95TPR"].values, color=colors)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(sub.index, rotation=45, ha="right")
        ax.set_ylabel("FPR @ 95% TPR  (lower is better)")
        ax.set_title(f"{ds_name} -- false alarms to catch 95% of zero-days "
                     f"({primary})")
        present = sorted({model_regime(m) for m in sub.index},
                         key=list(REGIME_COLORS).index)
        ax.legend(handles=[Patch(color=REGIME_COLORS[r], label=r)
                           for r in present], fontsize=8)
        savefig(fig, os.path.join(figdir, "zeroday_fpr95_bar.png"))

    # (c) AUROC & AUPR per novelty score (grouped bars) -----------------------
    for metric in ["AUROC", "AUPR"]:
        if metric not in det_df.columns:
            continue
        piv = det_df.pivot_table(index="Model", columns="Score",
                                 values=metric).reindex(order)
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.plot(kind="bar", ax=ax, width=0.8)
        ax.axhline(0.5 if metric == "AUROC" else piv.values[np.isfinite(
            piv.values)].min(), color="grey", lw=0.8, ls="--")
        ax.set_ylim(0, 1.02); ax.set_ylabel(f"Zero-day detection {metric}")
        ax.set_title(f"{ds_name} -- {metric} by novelty score and model")
        ax.legend(title="Score", fontsize=8)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        savefig(fig, os.path.join(figdir,
                                  f"zeroday_{metric.lower()}_by_score.png"))


# =============================================================================
# 7. EXPLAINABILITY: SHAP + LIME + 8 QUALITY METRICS
# =============================================================================
def _extract_class_attributions(shap_values, preds, n_classes):
    """Return (n, f) matrix of attributions for each instance's predicted class,
    handling the different return formats of the shap library."""
    if isinstance(shap_values, list):                      # old API: list per class
        return np.stack([np.asarray(shap_values[min(p, len(shap_values) - 1)])[i]
                         for i, p in enumerate(preds)])
    sv = np.asarray(shap_values)
    if sv.ndim == 3:                                       # (n, f, c) or (c, n, f)
        if sv.shape[-1] == n_classes:
            return np.stack([sv[i, :, p] for i, p in enumerate(preds)])
        if sv.shape[0] == n_classes:
            return np.stack([sv[p, i, :] for i, p in enumerate(preds)])
    return sv                                              # (n, f) binary case


def shap_explain(model_name, predict_proba, clf, X_bg, X_exp, n_classes, xcfg):
    """Returns per-instance predicted-class attributions + raw values + base."""
    preds = np.argmax(predict_proba(X_exp), axis=1)
    if model_name == "LightGBM":
        # TreeExplainer outputs follow the model's LOCAL class order
        # (clf.classes_), which can be smaller than the full label space.
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X_exp)
        base = explainer.expected_value
        n_local = len(clf.classes_)
        local = np.argmax(clf.predict_proba(X_exp), axis=1)
        A = _extract_class_attributions(raw, local, n_local)
        preds = np.asarray(clf.classes_, dtype=int)[local]   # global ids
        def explain_one(x):
            v = explainer.shap_values(x.reshape(1, -1))
            pl = int(np.argmax(clf.predict_proba(x.reshape(1, -1))))
            return _extract_class_attributions(v, [pl], n_local)[0]
        return np.asarray(A, dtype=float), raw, base, preds, explain_one
    else:
        bg = shap.kmeans(X_bg, min(25, len(X_bg)))
        explainer = shap.KernelExplainer(predict_proba, bg)
        raw = explainer.shap_values(X_exp, nsamples=xcfg["kernel_nsamples"],
                                    silent=True)
        base = explainer.expected_value
        def explain_one(x):
            v = explainer.shap_values(x.reshape(1, -1),
                                      nsamples=xcfg["kernel_nsamples"],
                                      silent=True)
            p = int(np.argmax(predict_proba(x.reshape(1, -1))))
            return _extract_class_attributions(v, [p], n_classes)[0]
    A = _extract_class_attributions(raw, preds, n_classes)
    return np.asarray(A, dtype=float), raw, base, preds, explain_one


def lime_explain(predict_proba, X_train_s, X_exp, feat_names, n_classes, xcfg,
                 class_names=None):
    """LIME tabular explanations for each instance's predicted class. Returns
    the (n, F) attribution matrix, the predictions, a single-instance explainer
    closure (for stability), and the fitted LimeTabularExplainer."""
    explainer = LimeTabularExplainer(
        X_train_s, feature_names=feat_names,
        class_names=(class_names if class_names is not None
                     else [str(c) for c in range(n_classes)]),
        discretize_continuous=False, random_state=SEED)
    preds = np.argmax(predict_proba(X_exp), axis=1)

    def explain_one(x):
        p = int(np.argmax(predict_proba(x.reshape(1, -1))))
        exp = explainer.explain_instance(
            x, predict_proba, labels=[p], num_features=len(feat_names),
            num_samples=xcfg["lime_num_samples"])
        vec = np.zeros(len(feat_names))
        for j, w in exp.as_map()[p]:
            vec[j] = w
        return vec

    A = np.stack([explain_one(X_exp[i]) for i in range(len(X_exp))])
    return A, preds, explain_one, explainer


def lime_instance_figures(explainer, predict_proba, X_exp, model_name,
                          figdir, xcfg, class_names=None, n_instances=2):
    """Per-instance LIME explanations: as_pyplot_figure PNG + interactive HTML."""
    for i in range(min(n_instances, len(X_exp))):
        try:
            x = X_exp[i]
            p = int(np.argmax(predict_proba(x.reshape(1, -1))))
            exp = explainer.explain_instance(
                x, predict_proba, labels=[p], num_features=12,
                num_samples=xcfg["lime_num_samples"])
            fig = exp.as_pyplot_figure(label=p)
            fig.set_size_inches(7, 5)
            cname = (class_names[p] if class_names is not None
                     and p < len(class_names) else str(p))
            fig.axes[0].set_title(
                f"LIME -- {model_name} -- instance {i} "
                f"(predicted: {cname})", fontsize=10)
            savefig(fig, os.path.join(
                figdir, f"lime_instance{i}_{model_name}.png"))
            exp.save_to_file(os.path.join(
                figdir, f"lime_instance{i}_{model_name}.html"))
        except Exception as e:
            log(f"    LIME instance figure {i} failed for {model_name}: {e}")


# ---- 8 explanation-quality metrics ------------------------------------------
def _prob_of_class(predict_proba, X, cls):
    """Vectorised helper: predicted probability of class `cls[i]` for row i."""
    P = predict_proba(np.atleast_2d(X))
    return P[np.arange(len(P)), cls]


def xai_fidelity(A, X, preds, predict_proba, baseline):
    """Spearman corr. between |attribution| and single-feature ablation drop."""
    cors = []
    p0 = _prob_of_class(predict_proba, X, preds)
    for i in range(len(X)):
        Xp = np.repeat(X[i:i + 1], X.shape[1], axis=0)
        Xp[np.arange(X.shape[1]), np.arange(X.shape[1])] = baseline
        drops = p0[i] - _prob_of_class(predict_proba, Xp,
                                       np.full(X.shape[1], preds[i]))
        r, _ = spearmanr(np.abs(A[i]), drops)
        if not np.isnan(r):
            cors.append(r)
    return float(np.mean(cors)) if cors else np.nan


def xai_infidelity(A, X, preds, predict_proba, xcfg):
    """Yeh et al. (2019): E[(d^T a - (f(x) - f(x-d)))^2]  (lower is better)."""
    rng = np.random.RandomState(SEED)
    vals = []
    p0 = _prob_of_class(predict_proba, X, preds)
    for i in range(len(X)):
        d = rng.normal(0, xcfg["infidelity_sigma"],
                       size=(xcfg["infidelity_perturbs"], X.shape[1]))
        fp = _prob_of_class(predict_proba, X[i] - d,
                            np.full(len(d), preds[i]))
        vals.append(np.mean((d @ A[i] - (p0[i] - fp)) ** 2))
    return float(np.mean(vals))


def xai_sparsity(A):
    """Gini index of |attributions| (1 = maximally sparse/concentrated)."""
    ginis = []
    for a in np.abs(A):
        a = np.sort(a)
        n, s = len(a), a.sum()
        if s == 0:
            continue
        ginis.append((2 * np.sum((np.arange(1, n + 1)) * a) / (n * s)) - (n + 1) / n)
    return float(np.mean(ginis)) if ginis else np.nan


def xai_stability(explain_one, X, xcfg):
    """Mean relative sensitivity of explanations to small input noise
    (lower is better)."""
    rng = np.random.RandomState(SEED)
    vals = []
    for i in range(min(xcfg["stability_instances"], len(X))):
        a0 = explain_one(X[i])
        for _ in range(xcfg["stability_perturbs"]):
            xp = X[i] + rng.normal(0, xcfg["stability_sigma"], size=X.shape[1])
            ap = explain_one(xp)
            vals.append(np.linalg.norm(a0 - ap) /
                        (np.linalg.norm(a0) + 1e-12))
    return float(np.mean(vals)) if vals else np.nan


def xai_comp_suff(A, X, preds, predict_proba, baseline, top_k_list):
    """Comprehensiveness (higher better) and Sufficiency (lower better)."""
    p0 = _prob_of_class(predict_proba, X, preds)
    comp, suff = [], []
    for k in top_k_list:
        Xc, Xs = X.copy(), np.tile(baseline, (len(X), 1))
        for i in range(len(X)):
            top = np.argsort(-np.abs(A[i]))[:k]
            Xc[i, top] = baseline[top]        # remove top-k
            Xs[i, top] = X[i, top]            # keep only top-k
        comp.append(p0 - _prob_of_class(predict_proba, Xc, preds))
        suff.append(p0 - _prob_of_class(predict_proba, Xs, preds))
    return float(np.mean(comp)), float(np.mean(suff))


def xai_agreement(A_shap, A_lime, k=10):
    """Mean per-instance top-k overlap between SHAP and LIME rankings."""
    aggr = []
    for a, b in zip(np.abs(A_shap), np.abs(A_lime)):
        ta = set(np.argsort(-a)[:k]); tb = set(np.argsort(-b)[:k])
        aggr.append(len(ta & tb) / k)
    return float(np.mean(aggr))


def xai_rank_correlation(A_shap, A_lime):
    """Spearman correlation of global mean-|attribution| rankings."""
    r, _ = spearmanr(np.abs(A_shap).mean(0), np.abs(A_lime).mean(0))
    return float(r)


# ---- SHAP/LIME figures -------------------------------------------------------
def shap_figures(model_name, raw, base, A, X_exp, feat_names, preds, figdir):
    """Save the four standard SHAP plots for one model: beeswarm, global bar,
    single-instance waterfall, and decision plot (each guarded independently)."""
    tag = model_name.replace("/", "_")
    try:                                                   # beeswarm
        vals = A
        fig = plt.figure(figsize=(8, 7))
        shap.summary_plot(vals, X_exp, feature_names=feat_names, show=False,
                          max_display=20)
        savefig(plt.gcf(), os.path.join(figdir, f"shap_beeswarm_{tag}.png"))
    except Exception as e:
        log(f"    beeswarm plot failed for {model_name}: {e}")
    try:                                                   # bar (global)
        fig = plt.figure(figsize=(8, 7))
        shap.summary_plot(A, X_exp, feature_names=feat_names, plot_type="bar",
                          show=False, max_display=20)
        savefig(plt.gcf(), os.path.join(figdir, f"shap_bar_{tag}.png"))
    except Exception as e:
        log(f"    bar plot failed: {e}")
    b0 = base
    if isinstance(base, (list, np.ndarray)):
        b0 = np.asarray(base).ravel()[min(int(preds[0]),
                                          np.asarray(base).size - 1)]
    try:                                                   # waterfall (inst 0)
        exp0 = shap.Explanation(values=A[0], base_values=float(b0),
                                data=X_exp[0], feature_names=feat_names)
        fig = plt.figure(figsize=(8, 7))
        shap.plots.waterfall(exp0, max_display=15, show=False)
        savefig(plt.gcf(), os.path.join(figdir, f"shap_waterfall_{tag}.png"))
    except Exception as e:
        log(f"    waterfall plot failed: {e}")
    try:                                                   # decision plot
        fig = plt.figure(figsize=(8, 7))
        shap.decision_plot(float(b0), A, feature_names=list(feat_names),
                           show=False)
        savefig(plt.gcf(), os.path.join(figdir, f"shap_decision_{tag}.png"))
    except Exception as e:
        log(f"    decision plot failed: {e}")


def lime_figure(A_lime, feat_names, model_name, figdir):
    """Global LIME importance bar chart: top-20 features by mean |weight|."""
    mean_abs = np.abs(A_lime).mean(0)
    order = np.argsort(mean_abs)[::-1][:20]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh([feat_names[i] for i in order][::-1], mean_abs[order][::-1],
            color="#2e8b57")
    ax.set_title(f"LIME global importance (mean |weight|) -- {model_name}")
    savefig(fig, os.path.join(figdir, f"lime_bar_{model_name}.png"))


def run_xai_for_model(model_name, predict_proba, clf, X_tr_s, X_te_s,
                      feat_names, n_classes, xcfg, figdir, class_names=None):
    """Full explainability pass for one model: compute SHAP and LIME
    attributions, save their figures, evaluate the 8 explanation-quality
    metrics (6 per-method + 2 cross-method), and return the metric rows plus
    the raw attribution matrices."""
    rng = np.random.RandomState(SEED)
    bg_idx = rng.choice(len(X_tr_s), min(xcfg["n_background"], len(X_tr_s)),
                        replace=False)
    ex_idx = rng.choice(len(X_te_s), min(xcfg["n_explain"], len(X_te_s)),
                        replace=False)
    X_bg, X_exp = X_tr_s[bg_idx], X_te_s[ex_idx]
    baseline = X_tr_s.mean(axis=0)
    rows, attributions = [], {}
    # clamp top-k sizes to the feature count (quantum models see only q
    # features, so k = 10/15 would otherwise remove/keep every feature)
    n_feat = X_tr_s.shape[1]
    xcfg = dict(xcfg)
    xcfg["top_k_list"] = sorted({min(k, max(1, n_feat - 1))
                                 for k in xcfg["top_k_list"]})
    k_agree = min(10, max(1, n_feat // 2))

    # ---------------- SHAP ----------------
    t0 = time.time()
    A_shap, raw, base, preds, shap_one = shap_explain(
        model_name, predict_proba, clf, X_bg, X_exp, n_classes, xcfg)
    shap_figures(model_name, raw, base, A_shap, X_exp, feat_names, preds, figdir)
    comp, suff = xai_comp_suff(A_shap, X_exp, preds, predict_proba,
                               baseline, xcfg["top_k_list"])
    rows.append({
        "Model": model_name, "Method": "SHAP",
        "Fidelity": xai_fidelity(A_shap, X_exp, preds, predict_proba, baseline),
        "Infidelity": xai_infidelity(A_shap, X_exp, preds, predict_proba, xcfg),
        "Sparsity": xai_sparsity(A_shap),
        "Stability": xai_stability(shap_one, X_exp, xcfg),
        "Comprehensiveness": comp, "Sufficiency": suff,
        "Time_s": time.time() - t0})
    attributions["SHAP"] = A_shap

    # ---------------- LIME ----------------
    t0 = time.time()
    A_lime, preds_l, lime_one, lime_explainer = lime_explain(
        predict_proba, X_tr_s[bg_idx], X_exp, feat_names, n_classes, xcfg,
        class_names=class_names)
    lime_figure(A_lime, feat_names, model_name, figdir)
    lime_instance_figures(lime_explainer, predict_proba, X_exp, model_name,
                          figdir, xcfg, class_names=class_names)
    comp, suff = xai_comp_suff(A_lime, X_exp, preds_l, predict_proba,
                               baseline, xcfg["top_k_list"])
    rows.append({
        "Model": model_name, "Method": "LIME",
        "Fidelity": xai_fidelity(A_lime, X_exp, preds_l, predict_proba, baseline),
        "Infidelity": xai_infidelity(A_lime, X_exp, preds_l, predict_proba, xcfg),
        "Sparsity": xai_sparsity(A_lime),
        "Stability": xai_stability(lime_one, X_exp, xcfg),
        "Comprehensiveness": comp, "Sufficiency": suff,
        "Time_s": time.time() - t0})
    attributions["LIME"] = A_lime

    # ---------------- cross-method metrics ----------------
    fa = xai_agreement(A_shap, A_lime, k=k_agree)
    rc = xai_rank_correlation(A_shap, A_lime)
    for r in rows:
        r["FeatureAgreement"] = fa
        r["RankCorrelation"] = rc
    return rows, attributions


# =============================================================================
# 8. REPORTING (per-dataset + cross-dataset comparison figures)
# =============================================================================
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


def dataset_summary_figures(ds_name, perf_df, xai_df, mean_abs_shap, feat_names,
                            figdir):
    """Per-dataset summary heatmaps: model x performance-metric, model|method x
    explanation-quality, and model x top-feature mean|SHAP|."""
    # performance heatmap (models x metrics)
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "Kappa"]
    heatmap_df(perf_df.set_index("Model")[metric_cols],
               f"{ds_name} -- model performance",
               os.path.join(figdir, "performance_heatmap.png"))
    # XAI quality heatmap
    xcols = ["Fidelity", "Infidelity", "Sparsity", "Stability",
             "Comprehensiveness", "Sufficiency", "FeatureAgreement",
             "RankCorrelation"]
    piv = xai_df.copy()
    piv["Model_Method"] = piv["Model"] + " | " + piv["Method"]
    heatmap_df(piv.set_index("Model_Method")[xcols],
               f"{ds_name} -- explanation quality",
               os.path.join(figdir, "xai_quality_heatmap.png"),
               cmap="magma")
    # global mean |SHAP| across models (models x top features)
    if mean_abs_shap:
        M = pd.DataFrame(mean_abs_shap).T
        M.columns = feat_names
        top = M.mean(0).sort_values(ascending=False).index[:15]
        heatmap_df(M[top], f"{ds_name} -- mean |SHAP| per model",
                   os.path.join(figdir, "shap_model_feature_heatmap.png"),
                   fmt=".2f", cmap="rocket" if HAS_SNS else "viridis")


def _ordered_pivot(df, index, columns, values, aggfunc="mean"):
    """Like df.pivot_table(...) but keeps index/columns in the order they
    first appear in df, instead of pivot_table's default alphabetical sort."""
    index_order = list(dict.fromkeys(df[index]))
    columns_order = list(dict.fromkeys(df[columns])) if isinstance(columns, str) \
        else None
    piv = df.pivot_table(index=index, columns=columns, values=values,
                          aggfunc=aggfunc)
    piv = piv.reindex(index=index_order)
    if columns_order is not None:
        piv = piv.reindex(columns=columns_order)
    return piv


def cross_dataset_figures(all_perf, all_xai, all_fs, all_zd, all_zdet, outdir):
    """Aggregate comparison figures across all processed datasets: per-metric
    model x dataset bars/heatmaps, timing heatmaps, few-shot vs zero-day
    episodic comparison, zero-day detection-rate and open-set AUROC heatmaps,
    and SHAP-vs-LIME explanation-quality summaries."""
    figdir = ensure_dir(os.path.join(outdir, "comparison_figures"))
    perf = pd.DataFrame(all_perf)
    xai = pd.DataFrame(all_xai)
    fs = pd.DataFrame(all_fs)
    zd = pd.DataFrame(all_zd)
    zdet = pd.DataFrame(all_zdet)        # stage-6C open-set detection metrics
    if perf.empty:
        return
    for metric in ["F1", "Accuracy", "AUC", "Kappa"]:
        piv = _ordered_pivot(perf, "Model", "Dataset", metric)
        fig, ax = plt.subplots(figsize=(10, 5))
        piv.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel(metric); ax.set_ylim(0, 1.02)
        ax.set_title(f"{metric} across datasets and models")
        ax.legend(fontsize=7, ncol=3)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        savefig(fig, os.path.join(figdir, f"compare_{metric}.png"))
        heatmap_df(piv, f"{metric}",
                   os.path.join(figdir, f"heatmap_{metric}.png"))

    # computational cost across datasets
    if "Train_time_s" in perf.columns:
        heatmap_df(_ordered_pivot(perf, "Model", "Dataset", "Train_time_s"),
                   "Training time (s)",
                   os.path.join(figdir, "heatmap_train_time.png"),
                   fmt=".1f", cmap="magma")
    if "XAI_time_s" in perf.columns:
        heatmap_df(_ordered_pivot(perf, "Model", "Dataset", "XAI_time_s"),
                   "XAI time (s, SHAP+LIME)",
                   os.path.join(figdir, "heatmap_xai_time.png"),
                   fmt=".1f", cmap="magma")

    # zero-day: episodic F1 heatmap + detection-rate comparison
    if not fs.empty and "zeroday_episodic_f1_macro" in fs.columns:
        heatmap_df(_ordered_pivot(fs, "Model", "Dataset",
                                  "zeroday_episodic_f1_macro"),
                   "Zero-day episodic F1 -- meta-learners x datasets",
                   os.path.join(figdir, "heatmap_zeroday_episodic_f1.png"),
                   cmap="rocket" if HAS_SNS else "viridis")
        # few-shot vs zero-day averaged over datasets
        model_order = list(dict.fromkeys(fs["Model"]))
        agg = fs.groupby("Model")[[c for c in
              ["fewshot_episodic_accuracy", "zeroday_episodic_accuracy",
               "fewshot_episodic_f1_macro", "zeroday_episodic_f1_macro"]
              if c in fs.columns]].mean().reindex(model_order)
        fig, ax = plt.subplots(figsize=(9, 5))
        agg.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylim(0, 1.02); ax.set_ylabel("Episodic score")
        ax.set_title("Few-shot vs zero-day episodic performance "
                     "(mean over datasets)")
        ax.legend(fontsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        savefig(fig, os.path.join(figdir, "fewshot_vs_zeroday_overall.png"))
    if not zd.empty:
        heatmap_df(_ordered_pivot(zd, "Model", "Dataset", "DetectionRate"),
                   "Zero-day detection rate (mean over held-out classes)",
                   os.path.join(figdir, "heatmap_zeroday_detection.png"),
                   cmap="rocket" if HAS_SNS else "viridis")

    # stage-6C open-set detection metrics (primary novelty score) across datasets
    if not zdet.empty:
        sub = zdet[zdet["Score"] == "MSP"]
        if not sub.empty:
            # AUROC and AUPR: higher is better ("rocket"/"viridis")
            for metric, cmap in [("AUROC", "rocket" if HAS_SNS else "viridis"),
                                 ("AUPR",  "rocket" if HAS_SNS else "viridis")]:
                if metric in sub.columns:
                    heatmap_df(
                        _ordered_pivot(sub, "Model", "Dataset", metric),
                        f"Zero-day open-set {metric} (MSP)",
                        os.path.join(figdir,
                                     f"heatmap_zeroday_detection_{metric.lower()}.png"),
                        cmap=cmap)
            # FPR@95%TPR: lower is better -> use a reversed colormap
            if "FPR@95TPR" in sub.columns:
                heatmap_df(
                    _ordered_pivot(sub, "Model", "Dataset", "FPR@95TPR"),
                    "Zero-day FPR@95%TPR (MSP; lower=better)",
                    os.path.join(figdir, "heatmap_zeroday_fpr95.png"),
                    cmap="rocket_r" if HAS_SNS else "viridis_r")

    plot_quantum_vs_classical(
        perf, zdet if not zdet.empty else None,
        "Quantum (Level A) vs classical models -- mean over datasets",
        os.path.join(figdir, "quantum_vs_classical_overall.png"))
    plot_level_comparison(
        perf, zdet if not zdet.empty else None,
        "Classical vs Level A vs Level B -- mean +- std over datasets",
        os.path.join(figdir, "level_comparison_overall.png"))

    if not xai.empty:
        for metric in ["Fidelity", "FeatureAgreement", "Stability",
                       "Comprehensiveness"]:
            dataset_order = list(dict.fromkeys(xai["Dataset"]))
            method_order = list(dict.fromkeys(xai["Method"]))
            piv = xai.pivot_table(index="Model", columns=["Dataset", "Method"],
                                  values=metric)
            model_order = list(dict.fromkeys(xai["Model"]))
            piv = piv.reindex(index=model_order)
            col_order = [(d, m) for d in dataset_order for m in method_order
                         if (d, m) in piv.columns]
            piv = piv.reindex(columns=col_order)
            heatmap_df(piv, f"{metric} -- explanation methods across datasets",
                       os.path.join(figdir, f"xai_heatmap_{metric}.png"),
                       cmap="magma")
        # SHAP vs LIME aggregate comparison
        method_order = list(dict.fromkeys(xai["Method"]))
        agg = xai.groupby("Method")[["Fidelity", "Infidelity", "Sparsity",
                                     "Stability", "Comprehensiveness",
                                     "Sufficiency"]].mean().reindex(method_order)
        fig, ax = plt.subplots(figsize=(9, 5))
        agg.T.plot(kind="bar", ax=ax)
        ax.set_title("SHAP vs LIME -- average explanation quality (all datasets)")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        savefig(fig, os.path.join(figdir, "shap_vs_lime_overall.png"))


# =============================================================================
# MAIN PER-DATASET DRIVER
# =============================================================================
def run_dataset(ds_name, dcfg, cfg):
    """End-to-end pipeline for a single dataset: load -> split -> select
    features -> balance -> train (base + meta) -> evaluate -> explain (XAI)
    -> report. Returns the five row-lists (performance, XAI, few-shot/zero-day
    episodic, zero-day per-class report, zero-day open-set detection) that
    main() aggregates across datasets."""
    set_seed()
    outdir = ensure_dir(os.path.join(cfg["output_dir"], ds_name))
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    xaidir = ensure_dir(os.path.join(outdir, "xai"))

    # 1-2. load / clean / split / scale ---------------------------------------
    X_df, y, le = load_and_clean(ds_name, dcfg, cfg)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), scaler = \
        split_and_scale(X_df, y, cfg)

    # 3. feature selection ------------------------------------------------------
    X_tr, X_va, X_te, feat_names, _ = select_features(
        X_tr, y_tr, X_va, X_te, list(X_df.columns), cfg["k_best"])

    # 4. class balancing (train only) ------------------------------------------
    X_tr, y_tr, kept = balance_classes(X_tr, y_tr, cfg)
    # keep val/test consistent with whichever classes survived balancing
    keep_mask_va = np.isin(y_va, kept); X_va, y_va = X_va[keep_mask_va], y_va[keep_mask_va]
    keep_mask_te = np.isin(y_te, kept); X_te, y_te = X_te[keep_mask_te], y_te[keep_mask_te]
    n_classes = len(le.classes_)
    class_names = [str(c) for c in le.classes_]

    # zero-day held-out classes (rarest non-benign train classes) ---------------
    # These classes are excluded from meta-training entirely (meta_pool) so
    # the meta-learners never see them during episodic training; they're
    # later evaluated purely via few-shot global/episodic adaptation to
    # simulate detecting a brand-new ("zero-day") attack type.
    mcfg = cfg["meta"]
    train_counts = Counter(y_tr)
    benign_ids = [i for i, c in enumerate(le.classes_)
                  if any(k in str(c).lower() for k in ("benign", "normal"))]
    candidates = sorted([c for c in kept if c not in benign_ids],
                        key=lambda c: train_counts[c])   # rarest first
    zero_day = candidates[:mcfg["zero_day_n_classes"]] if mcfg["zero_day"] else []
    meta_pool = np.array([c for c in kept if c not in zero_day])
    if len(meta_pool) < 2:                      # safety fallback
        meta_pool, zero_day = np.array(kept), []
    log(f"[{ds_name}] meta-train classes: {len(meta_pool)}  "
        f"zero-day held-out: {[class_names[c] for c in zero_day]}")

    perf_rows, xai_rows, fewshot_rows = [], [], []
    mean_abs_shap = {}
    models = {}
    train_times, xai_times = {}, {}
    meta_histories = {}
    meta_state = {}    # algo -> (params, lr_dict); reused for known-only re-adapt (stage 6C)

    # 5A. base learners ---------------------------------------------------------     
    log(f"[{ds_name}] ----------- Training -----------")
    log(f"[{ds_name}] training LightGBM ...")
    lgbm, lgbm_tr_time  = train_lightgbm(X_tr, y_tr, X_va, y_va)
    train_times["LightGBM"] = lgbm_tr_time
    models["LightGBM"] = (make_full_proba(lgbm, n_classes), lgbm)
    plot_lgbm_importance(lgbm, feat_names,
                         os.path.join(figdir, "lgbm_feature_importance.png"))

    log(f"[{ds_name}] training CNN ...")
    cnn, cnn_tr_time, cnn_hist = train_cnn(X_tr, y_tr, X_va, y_va,
                                           n_classes, cfg["cnn"])
    train_times["CNN"] = cnn_tr_time
    plot_cnn_loss(cnn_hist, ds_name, os.path.join(figdir, "loss_cnn.png"))
    cnn_w = TorchWrapper(cnn, n_classes)
    models["CNN"] = (cnn_w.predict_proba, None)    
    print("\t")

    # 5B. meta-learners ---------------------------------------------------------
    for algo in META_MODELS:
        log(f"[{ds_name}] meta-training {algo} ...")
        # (1) episodic meta-training on meta_pool classes only (zero-day excluded)
        params, lr_dict, algo_tr_time, algo_hist = meta_train(
            algo, X_tr, y_tr, meta_pool, X_tr.shape[1], n_classes, mcfg)
        train_times[algo] = algo_tr_time
        meta_histories[algo] = algo_hist
        # keep the meta-learned initialisation so stage 6C can re-adapt it to
        # the known-only label space without repeating meta-training
        meta_state[algo] = (params, lr_dict)
        # (2) adapt the meta-init to the FULL label space for full-test-set eval
        wrapper = global_adapt(algo, params, lr_dict, X_tr, y_tr,
                               n_classes, mcfg)
        models[algo] = (wrapper.predict_proba, None)

        # (3) separately measure "true" episodic few-shot performance on
        #     meta_pool classes, and (if enabled) zero-day performance on the
        #     held-out classes (+ one benign class so episodes are 2+ way)
        fs = episodic_eval(algo, params, lr_dict, X_te, y_te,
                           list(meta_pool), mcfg, tag="fewshot")
        zd = (episodic_eval(algo, params, lr_dict, X_te, y_te,
                            zero_day + benign_ids[:1], mcfg, tag="zeroday")
              if zero_day else {})
        fewshot_rows.append({"Dataset": ds_name, "Model": algo, **fs, **zd})
    plot_meta_losses(meta_histories, ds_name, figdir)
    save_json({a: h for a, h in meta_histories.items()},
              os.path.join(outdir, "meta_loss_histories.json"))
    save_json({"cnn": cnn_hist}, os.path.join(outdir, "cnn_loss_history.json"))

    # 5C. Level-A quantum models (QSVC, VQC, QNN, HQCNN) + q-feature controls --
    qcfg = cfg["quantum"]
    q_reducer, q_wrappers, eval_pp = None, {}, {}
    if qcfg.get("enabled", True) and not HAS_PL:
        log(f"[{ds_name}] PennyLane not installed -> quantum models skipped "
            f"(pip install pennylane)")
    elif qcfg.get("enabled", True):
        # 3Q. extra dimensionality reduction for quantum: top-q of the 30
        n_kept = len(np.unique(y_tr))
        q = max(int(qcfg["n_qubits"]), int(np.ceil(np.log2(max(n_kept, 2)))))
        if q != qcfg["n_qubits"]:
            log(f"[{ds_name}] q raised {qcfg['n_qubits']} -> {q} so 2^q >= "
                f"{n_kept} classes (VQC readout)")
        q_reducer = QFeatureReducer(X_tr, y_tr, feat_names, q)
        log(f"[{ds_name}] quantum features (q={q}): {q_reducer.names}")
        log(f"[{ds_name}] training Level-A quantum models ...")
        qsuite = train_quantum_suite(X_tr, y_tr, X_va, y_va, q_reducer,
                                     n_classes, qcfg)
        q_histories, q_info = {}, []
        for qname, (qw, qt, qhist) in qsuite.items():
            train_times[qname] = qt
            models[qname] = (qw.predict_proba, None)
            if qcfg.get("eval_shots"):
                eval_pp[qname] = qw.predict_proba_eval
            q_wrappers[qname] = qw
            if qhist:
                q_histories[qname] = qhist
            n_circ, n_tot = quantum_n_params(qw)
            q_info.append({"Model": qname, "Regime": model_regime(qname),
                           "Qubits": q, "Circuit_params": n_circ,
                           "Total_params": n_tot,
                           "Train_time_s": round(qt, 3)})
            log(f"[{ds_name}] {qname:6s} trained in {qt:.1f}s "
                f"(circuit params={n_circ}, total={n_tot})")
        plot_quantum_losses(q_histories, ds_name,
                            os.path.join(figdir, "loss_quantum.png"))
        plot_quantum_inputs(q_reducer, X_tr, y_tr, class_names,
                            qsuite["QSVC"][0].extra, ds_name, figdir)
        save_circuit_drawings(q_reducer, qcfg, len(np.unique(y_tr)),
                              os.path.join(outdir, "quantum_circuits.txt"))
        save_json({a: h for a, h in q_histories.items()},
                  os.path.join(outdir, "quantum_loss_histories.json"))
        save_json({"n_qubits": q, "features": q_reducer.names,
                   "f_scores": q_reducer.scores.tolist(),
                   "angle_lo": q_reducer.lo.tolist(),
                   "angle_hi": q_reducer.hi.tolist(),
                   "config": qcfg, "models": q_info},
                  os.path.join(outdir, "quantum_summary.json"))
        print("\t")

    # 5D. Level-B hybrid quantum prototypical networks -----------------------
    bcfg = cfg["quantum_b"]
    qb_nets = {}
    if bcfg.get("enabled", True) and HAS_PL and len(meta_pool) >= 2:
        if q_reducer is None:                 # Level A disabled -> own reducer
            q_reducer = QFeatureReducer(X_tr, y_tr, feat_names,
                                        int(qcfg["n_qubits"]))
            log(f"[{ds_name}] quantum features (q={q_reducer.q}): "
                f"{q_reducer.names}")
        log(f"[{ds_name}] meta-training Level-B hybrid quantum ProtoNets "
            f"(q={q_reducer.q}, d={bcfg['embed_dim']}, "
            f"{min(bcfg['n_way'], len(meta_pool))}-way {bcfg['k_shot']}-shot) ...")
        Xa_tr = q_reducer.angles(q_reducer.reduce(X_tr))
        Xa_te = q_reducer.angles(q_reducer.reduce(X_te))
        Xs_g, ys_g = support_set(X_tr, y_tr, bcfg["adapt_shot"])   # all classes
        names_b = QUANTUM_B_MODELS + (["MLP-PN"] if bcfg.get("classical_control",
                                                             True) else [])
        b_hist, b_info, b_wrappers = {}, [], {}
        for bname in names_b:
            # (1) episodic meta-training on meta_pool only (zero-day excluded)
            net, bt, bh = meta_train_qproto(bname, Xa_tr, y_tr, meta_pool,
                                            q_reducer.q, bcfg)
            qb_nets[bname], b_hist[bname] = net, bh
            train_times[bname] = bt
            # (2) global prototypes from adapt_shot rows of EVERY train class
            w = QProtoWrapper(bname, net, q_reducer, Xs_g, ys_g, n_classes, bcfg)
            models[bname] = (w.predict_proba, None)
            q_wrappers[bname] = b_wrappers[bname] = w
            if bcfg.get("eval_shots"):
                eval_pp[bname] = w.predict_proba_eval
            # (3) episodic few-shot (meta_pool) and zero-day (held-out) tests
            fs = qproto_episodic_eval(net, Xa_te, y_te, list(meta_pool), bcfg,
                                      "fewshot")
            zd = (qproto_episodic_eval(net, Xa_te, y_te,
                                       list(zero_day) + benign_ids[:1], bcfg,
                                       "zeroday") if zero_day else {})
            fewshot_rows.append({"Dataset": ds_name, "Model": bname, **fs, **zd})
            n_circ, n_tot = qproto_n_params(net)
            b_info.append({"Model": bname, "Regime": model_regime(bname),
                           "Circuit_params": n_circ, "Total_params": n_tot,
                           "Embed_dim": int(net.d), "Train_time_s": round(bt, 2),
                           **fs, **zd})
            log(f"[{ds_name}] {bname:9s} meta-trained in {bt:.1f}s  "
                f"few-shot acc={fs.get('fewshot_episodic_accuracy', np.nan):.3f}  "
                f"zero-day acc={zd.get('zeroday_episodic_accuracy', np.nan):.3f}")
        plot_levelB_losses(b_hist, ds_name, os.path.join(figdir, "loss_levelB.png"))
        plot_levelB_embeddings(b_wrappers, X_te, y_te, class_names, zero_day,
                               ds_name, figdir)
        save_levelB_circuits(qb_nets, q_reducer,
                             os.path.join(outdir, "levelB_circuits.txt"))
        save_json(b_hist, os.path.join(outdir, "levelB_loss_histories.json"))
        save_json({"features": q_reducer.names, "config": bcfg,
                   "models": b_info},
                  os.path.join(outdir, "levelB_summary.json"))
        print("\t")

    # 6. evaluation --------------------------------------------------------------    
    log(f"[{ds_name}] ----------- Evaluation -----------")
    for name, (pp, clf) in models.items():
        t0 = time.time()
        proba = eval_pp.get(name, pp)(X_te)    # inference only (shots if set)
        eval_time = time.time() - t0
        proba = sanitize_proba(proba)          # guard: no NaN/inf into metrics/plots
        pred = np.argmax(proba, axis=1)
        m = compute_metrics(y_te, pred, proba, n_classes)
        B = cfg.get("bootstrap", 0)
        m["Accuracy_lo"], m["Accuracy_hi"] = bootstrap_ci(accuracy_score,
                                                          y_te, pred, B=B)
        m["F1_lo"], m["F1_hi"] = bootstrap_ci(
            lambda a, b: f1_score(a, b, average="macro", zero_division=0),
            y_te, pred, B=B)
        m.update({"Dataset": ds_name, "Model": name,
                  "Regime": model_regime(name),
                  "Train_time_s": round(train_times.get(name, np.nan), 3),
                  "Eval_time_s": round(eval_time, 4),
                  "Eval_ms_per_1k": round(1000.0 * eval_time /
                                          max(len(X_te), 1) * 1000, 3)})
        perf_rows.append(m)
        log(f"[{ds_name}] {name:9s}  Acc={m['Accuracy']:.4f} "
            f"F1={m['F1']:.4f}  AUC={m['AUC']:.4f}  Kappa={m['Kappa']:.4f}  "
            f"train={m['Train_time_s']:.1f}s  eval={eval_time:.2f}s")
        plot_confusion(y_te, pred, class_names, f"{ds_name} -- {name}",
                       os.path.join(figdir, f"cm_{name}.png"), normalize=False)
        plot_confusion(y_te, pred, class_names, f"{ds_name} -- {name}",
                       os.path.join(figdir, f"cm-norm_{name}.png"))
        
        plot_roc(y_te, proba, class_names, f"{ds_name} -- {name} ROC",
                 os.path.join(figdir, f"roc_{name}.png"))

    # 6b. zero-day global detection statistics (full test set) --------------------
    zeroday_rows = (zeroday_global_report(ds_name, models, X_te, y_te,
                                          zero_day, class_names)
                    if zero_day else [])

    # 6c. unified zero-day (open-set / novelty) detection for ALL models ----------
    #     Re-fits/re-adapts every model on the known classes only, then scores
    #     the held-out zero-day classes as an open-set detection problem.
    log(f"[{ds_name}] ----------- Zero-day detection (open-set) -----------")
    zeroday_detect_rows = run_zeroday_detection(
        ds_name, meta_state, X_tr, y_tr, X_va, y_va, X_te, y_te,
        meta_pool, zero_day, n_classes, class_names, cfg, figdir, outdir,
        q_reducer=q_reducer, qb_nets=qb_nets)

    # 7. explainability ----------------------------------------------------------- 
    log(f"[{ds_name}] ----------- Explainability -----------")
    for name, (pp, clf) in models.items():
        log(f"[{ds_name}] XAI (SHAP + LIME) for {name} ...")
        t0 = time.time()
        try:
            if name in q_wrappers:
                # quantum models / q-controls are explained in the q-feature
                # space they actually receive (q features, not 30)
                qw = q_wrappers[name]
                rows, attr = run_xai_for_model(
                    name, qw.predict_proba_q, None,
                    q_reducer.reduce(X_tr), q_reducer.reduce(X_te),
                    q_reducer.names, n_classes, cfg["xai"], xaidir,
                    class_names=class_names)
                full = np.zeros(len(feat_names))           # scatter q -> 30
                full[q_reducer.idx] = np.abs(attr["SHAP"]).mean(0)
                for r in rows:
                    r["Feature_space"] = f"q={q_reducer.q}"
            else:
                rows, attr = run_xai_for_model(name, pp, clf, X_tr, X_te,
                                               feat_names, n_classes,
                                               cfg["xai"], xaidir,
                                               class_names=class_names)
                full = np.abs(attr["SHAP"]).mean(0)
                for r in rows:
                    r["Feature_space"] = f"k={len(feat_names)}"
            for r in rows:
                r["Dataset"] = ds_name
            xai_rows.extend(rows)
            mean_abs_shap[name] = full
            np.save(os.path.join(xaidir, f"shap_values_{name}.npy"),
                    attr["SHAP"])
            np.save(os.path.join(xaidir, f"lime_values_{name}.npy"),
                    attr["LIME"])
        except Exception as e:
            log(f"[{ds_name}] XAI failed for {name}: {e}")
        xai_times[name] = time.time() - t0
    for m in perf_rows:                       # merge total XAI time per model
        m["XAI_time_s"] = round(xai_times.get(m["Model"], np.nan), 2)

    # 8. per-dataset reporting -----------------------------------------------------
    log(f"[{ds_name}] ----------- Reporting -----------")
    perf_df = pd.DataFrame(perf_rows)
    xai_df = pd.DataFrame(xai_rows)
    fs_df = pd.DataFrame(fewshot_rows)
    zd_df = pd.DataFrame(zeroday_rows)
    perf_df.to_csv(os.path.join(outdir, "model_metrics.csv"), index=False)
    xai_df.to_csv(os.path.join(outdir, "xai_metrics.csv"), index=False)
    fs_df.to_csv(os.path.join(outdir, "fewshot_zeroday_metrics.csv"), index=False)
    if not zd_df.empty:
        zd_df.to_csv(os.path.join(outdir, "zeroday_class_report.csv"),
                     index=False)
    zdet_df = pd.DataFrame(zeroday_detect_rows)          # stage-6C open-set metrics
    if not zdet_df.empty:
        zdet_df.to_csv(os.path.join(outdir, "zeroday_detection_metrics.csv"),
                       index=False)
    perf_df[["Model", "Train_time_s", "Eval_time_s", "Eval_ms_per_1k",
             "XAI_time_s"]].to_csv(os.path.join(outdir, "timing.csv"),
                                   index=False)
    plot_time_comparison(perf_df, ds_name,
                         os.path.join(figdir, "time_comparison.png"))
    plot_zeroday_figures(zd_df if not zd_df.empty else None, fs_df,
                         ds_name, figdir)
    plot_quantum_vs_classical(
        perf_df, zdet_df if not zdet_df.empty else None,
        f"{ds_name} -- quantum (Level A) vs classical models",
        os.path.join(figdir, "quantum_vs_classical.png"))
    plot_level_comparison(
        perf_df, zdet_df if not zdet_df.empty else None,
        f"{ds_name} -- Classical vs Level A vs Level B",
        os.path.join(figdir, "level_comparison.png"))
    save_json({"dataset": ds_name, "n_features": len(feat_names),
               "features": feat_names, "n_classes": n_classes,
               "classes": class_names,
               "zero_day_classes": [class_names[c] for c in zero_day]},
              os.path.join(outdir, "dataset_summary.json"))
    if not xai_df.empty:
        dataset_summary_figures(ds_name, perf_df, xai_df, mean_abs_shap,
                                feat_names, figdir)
    log(f"[{ds_name}] DONE -> {outdir}")
    return perf_rows, xai_rows, fewshot_rows, zeroday_rows, zeroday_detect_rows


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    """CLI entry point: parse args, build the effective config, run every
    requested dataset through run_dataset(), then produce cross-dataset
    aggregate CSVs + comparison figures."""
    ap = argparse.ArgumentParser(
        description="Explainable Meta-Learning for Few-Shot and Zero-Day "
                    "Intrusion Detection in IoT Networks")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                    choices=list(DATASETS.keys()))
    ap.add_argument("--episodes", type=int, default=None,
                    help="override meta-training episodes")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="override the stratified row cap")
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke-test configuration")
    ap.add_argument("--qubits", type=int, default=None,
                    help="number of qubits / quantum features q (4-8)")
    ap.add_argument("--eval-shots", type=int, default=None,
                    help="finite-shot quantum evaluation (default: exact)")
    ap.add_argument("--no-quantum", action="store_true",
                    help="disable all quantum models (Level A and B)")
    ap.add_argument("--level", choices=["A", "B", "AB"], default="AB",
                    help="which quantum extension(s) to run")
    ap.add_argument("--embed-dim", type=int, default=None,
                    help="Level-B embedding dimension d (8-32)")
    args = ap.parse_args()

    # start from the global default CONFIG, then apply CLI overrides
    cfg = copy.deepcopy(CONFIG)
    if args.episodes:
        cfg["meta"]["episodes"] = args.episodes
    if args.max_rows:
        cfg["max_rows"] = args.max_rows
    if args.output:
        cfg["output_dir"] = args.output
    if args.quick:
        # aggressively shrink every expensive knob for a fast smoke test
        cfg["max_rows"] = 30_000
        cfg["meta"].update(episodes=60, eval_episodes=20, adapt_steps=30)
        cfg["cnn"].update(epochs=8)
        cfg["xai"].update(n_explain=15, kernel_nsamples=100,
                          lime_num_samples=500, stability_instances=3,
                          stability_perturbs=2)
        cfg["quantum"].update(qsvc_per_class=25, train_per_class=60,
                              val_per_class=20, epochs=6, layers=2)
        cfg["quantum_b"].update(episodes=60, eval_episodes=30, layers=2)
        cfg["bootstrap"] = 50
    if args.qubits:
        cfg["quantum"]["n_qubits"] = args.qubits
    if args.eval_shots:
        cfg["quantum"]["eval_shots"] = args.eval_shots
        cfg["quantum_b"]["eval_shots"] = args.eval_shots
    if args.embed_dim:
        cfg["quantum_b"]["embed_dim"] = args.embed_dim
    cfg["quantum"]["enabled"] &= "A" in args.level
    cfg["quantum_b"]["enabled"] &= "B" in args.level
    if args.no_quantum:
        cfg["quantum"]["enabled"] = False
        cfg["quantum_b"]["enabled"] = False

    ensure_dir(cfg["output_dir"])
    set_seed()
    log(f"device: {DEVICE}")

    # run each dataset independently; one dataset failing doesn't abort the rest
    all_perf, all_xai, all_fs, all_zd, all_zdet = [], [], [], [], []
    for ds in args.datasets:
        log(f"===== {ds} =====")
        try:
            p, x, f, z, zdet = run_dataset(ds, DATASETS[ds], cfg)
            all_perf += p; all_xai += x; all_fs += f; all_zd += z
            all_zdet += zdet
        except FileNotFoundError as e:
            log(str(e))                     # missing CSV -> skip, don't crash
        except Exception as e:
            import traceback
            log(f"[{ds}] FAILED: {e}")
            traceback.print_exc()

    # cross-dataset aggregation: combined CSVs + comparison heatmaps/bars
    if all_perf:
        pd.DataFrame(all_perf).to_csv(
            os.path.join(cfg["output_dir"], "all_model_metrics.csv"), index=False)
        pd.DataFrame(all_xai).to_csv(
            os.path.join(cfg["output_dir"], "all_xai_metrics.csv"), index=False)
        pd.DataFrame(all_fs).to_csv(
            os.path.join(cfg["output_dir"], "all_fewshot_zeroday.csv"), index=False)
        if all_zd:
            pd.DataFrame(all_zd).to_csv(
                os.path.join(cfg["output_dir"], "all_zeroday_class_report.csv"),
                index=False)
        if all_zdet:
            pd.DataFrame(all_zdet).to_csv(
                os.path.join(cfg["output_dir"], "all_zeroday_detection.csv"),
                index=False)
        cross_dataset_figures(all_perf, all_xai, all_fs, all_zd, all_zdet,
                              cfg["output_dir"])
        log(f"ALL DONE -> {cfg['output_dir']}/")


if __name__ == "__main__":
    main()