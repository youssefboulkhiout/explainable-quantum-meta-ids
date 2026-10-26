"""Per-dataset driver: runs stages 1-8 for one dataset.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .config import META_MODELS, QUANTUM_B_MODELS
from .utils import ensure_dir, log, sanitize_proba, save_json, set_seed
from .data import balance_classes, load_and_clean, select_features, split_and_scale
from .baselines import TorchWrapper, make_full_proba, train_cnn, train_lightgbm
from .meta_learning import episodic_eval, global_adapt, meta_train
from .quantum_a import HAS_PL, QFeatureReducer, model_regime, plot_quantum_inputs, plot_quantum_losses, plot_quantum_vs_classical, quantum_n_params, save_circuit_drawings, train_quantum_suite
from .quantum_b import QProtoWrapper, bootstrap_ci, meta_train_qproto, plot_levelB_embeddings, plot_levelB_losses, plot_level_comparison, qproto_episodic_eval, qproto_n_params, save_levelB_circuits, support_set
from .evaluation import compute_metrics, plot_cnn_loss, plot_confusion, plot_lgbm_importance, plot_meta_losses, plot_roc, plot_time_comparison, plot_zeroday_figures, zeroday_global_report
from .zeroday import run_zeroday_detection
from .xai import run_xai_for_model
from .reporting import dataset_summary_figures

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
