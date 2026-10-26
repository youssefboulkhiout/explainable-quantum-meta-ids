"""Command-line entry point (same arguments as the original script).

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import copy
import argparse
import pandas as pd

from .config import CONFIG, DATASETS, DEVICE
from .utils import ensure_dir, log, set_seed
from .reporting import cross_dataset_figures
from .pipeline import run_dataset

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
