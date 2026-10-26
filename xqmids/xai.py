"""Stage 7: SHAP + LIME explanations and the 8 explanation-quality metrics.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import time
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from .config import SEED
from .utils import log, savefig

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
