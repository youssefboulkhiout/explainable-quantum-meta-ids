"""Stage 6C: unified open-set zero-day detection (MSP / entropy / margin) for every model.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc

from .config import ALL_MODELS, SEED
from .utils import heatmap_df, log, savefig
from .baselines import TorchWrapper, make_full_proba, train_cnn, train_lightgbm
from .meta_learning import global_adapt
from .quantum_a import REGIME_COLORS, model_regime, train_quantum_suite
from .quantum_b import QProtoWrapper, bootstrap_ci, support_set

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
