"""Stages 1-4: loading/cleaning, 70/10/20 split + scaling, SelectKBest, class balancing.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from .config import GENERIC_DROP_EXACT, SEED
from .utils import log

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
