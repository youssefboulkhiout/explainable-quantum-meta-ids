"""Stage 8: per-dataset and cross-dataset reporting figures.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import pandas as pd
import matplotlib.pyplot as plt

from ._env import HAS_SNS
from .utils import ensure_dir, heatmap_df, savefig
from .quantum_a import plot_quantum_vs_classical
from .quantum_b import plot_level_comparison

# =============================================================================
# 8. REPORTING (per-dataset + cross-dataset comparison figures)
# =============================================================================


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
