# -*- coding: utf-8 -*-
"""
run_in_spyder.py -- launch the explainable-quantum-meta-ids benchmark from
Spyder (or any IDE) by pressing F5.

Edit the OPTIONS block below, then run the file. It builds the same
command-line arguments as

    python -m xqmids [options]

and calls the package entry point, so the results are identical to a
terminal run with the same options. All parameters of the paper are the
defaults in xqmids/config.py; leave every option at its default value to
reproduce the published results.

Expected layout (paths are relative to this file):
    data/ASEADOS_SDN_IoT.csv
    data/RT-IoT2022.csv
    data/UNSW-NB15.csv
Outputs are written to outputs/ (or to OUTPUT below).
"""
import os
import sys

# =============================================================================
# OPTIONS (None / False = paper default)
# =============================================================================
DATASETS = ["ASEADOS_SDN_IoT", "RT-IoT2022", "UNSW-NB15"]  # any subset
LEVEL = "AB"          # quantum extension(s): "A", "B" or "AB" (paper: "AB")
QUICK = False         # True = fast smoke test (NOT the paper configuration)
NO_QUANTUM = False    # True = classical models only
EPISODES = None       # override meta-training episodes (paper: 500)
MAX_ROWS = None       # override the stratified row cap (paper: 200000)
QUBITS = None         # number of qubits / quantum features q (paper: 4)
EVAL_SHOTS = None     # finite-shot quantum evaluation (paper: exact, None)
EMBED_DIM = None      # Level-B embedding dimension d (paper: 16)
OUTPUT = None         # output folder (paper: "outputs")
N_THREADS = None      # e.g. 4 to limit CPU threads; None = library default
# =============================================================================


def build_argv():
    """Translate the OPTIONS block into the package's command-line arguments."""
    argv = ["xqmids", "--datasets", *DATASETS, "--level", LEVEL]
    if QUICK:
        argv.append("--quick")
    if NO_QUANTUM:
        argv.append("--no-quantum")
    for flag, value in (("--episodes", EPISODES), ("--max-rows", MAX_ROWS),
                        ("--qubits", QUBITS), ("--eval-shots", EVAL_SHOTS),
                        ("--embed-dim", EMBED_DIM), ("--output", OUTPUT)):
        if value is not None:
            argv += [flag, str(value)]
    return argv


if __name__ == "__main__":
    # run from the repository root so that data/ and outputs/ resolve the
    # same way as in a terminal run
    ROOT = os.path.dirname(os.path.abspath(__file__))
    os.chdir(ROOT)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    if N_THREADS:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(N_THREADS)

    missing = [d for d in DATASETS
               if not os.path.exists(os.path.join(ROOT, "data", f"{d}.csv"))]
    if missing:
        print("WARNING: missing dataset file(s): "
              + ", ".join(f"data/{d}.csv" for d in missing)
              + " -- see data/README.md. They will be skipped.")

    from xqmids.cli import main

    sys.argv = build_argv()
    print("Running:", " ".join(["python -m"] + sys.argv))
    main()
