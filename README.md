# explainable-quantum-meta-ids

Code for the paper **"Explainable Quantum-Enhanced Meta-Learning for Few-Shot
and Zero-Day Intrusion Detection in IoT Networks"** (Y. Boulkhiout,
S. Balbal, L. Goudjil, K. Nasri).

The benchmark compares, on ASEADOS-SDN-IoT, RT-IoT2022 and UNSW-NB15:

| Family | Models |
|---|---|
| Supervised baselines | LightGBM, 1-D CNN |
| Classical meta-learners | Reptile, MAML, FOMAML, Meta-SGD, ANIL, BOIL, ProtoNet |
| Level A quantum models | QSVC, VQC, QNN, HQCNN |
| Level B quantum prototypical networks | QSVC-PN, VQC-PN, QNN-PN, HQCNN-PN |
| Matched classical controls (same q features) | SVM-q, MLP-q, MLP-PN |

All models are evaluated on closed-set classification, episodic few-shot and
zero-day tasks (the two rarest attack families are withheld from training),
and open-set zero-day detection (MSP, entropy, margin novelty scores), and
are explained with SHAP and LIME using eight explanation-quality metrics
(fidelity, infidelity, sparsity, stability, comprehensiveness, sufficiency,
feature agreement, rank correlation). Quantum circuits are simulated exactly
with PennyLane on CPU.

## Repository structure

```
explainable-quantum-meta-ids/
├── run_in_spyder.py        # launcher for Spyder / any IDE (edit options, press F5)
├── xqmids/                 # the pipeline, split by stage
│   ├── config.py           # SEED, CONFIG (all paper parameters), dataset paths, model lists
│   ├── _env.py             # headless Matplotlib backend, seaborn theme, warning filter
│   ├── utils.py            # seeding, logging, saving, probability guard, plot helpers
│   ├── data.py             # 1-4: loading/cleaning, 70/10/20 split, SelectKBest, SMOTE balancing
│   ├── baselines.py        # 5A: LightGBM, 1-D CNN
│   ├── meta_learning.py    # 5B: the seven classical meta-learners
│   ├── quantum_a.py        # 3Q/5C: q-feature reduction, QSVC, VQC, QNN, HQCNN, SVM-q, MLP-q
│   ├── quantum_b.py        # 5D: quantum prototypical networks, MLP-PN, bootstrap CIs
│   ├── evaluation.py       # 6/6b: metrics, confusion matrices, ROC, per-class zero-day report
│   ├── zeroday.py          # 6C: open-set zero-day detection for every model
│   ├── xai.py              # 7: SHAP, LIME and the eight quality metrics
│   ├── reporting.py        # 8: per-dataset and cross-dataset figures
│   ├── pipeline.py         # run_dataset(): stages 1-8 for one dataset
│   ├── cli.py              # main(): command-line options and cross-dataset aggregation
│   └── __main__.py         # enables `python -m xqmids`
├── data/                   # put the three CSV files here (see data/README.md)
├── reference/
│   └── xai_meta_qml_levelAB.py   # original single-file script (for comparison)
├── scripts/
│   └── run_quick_test.sh   # smoke test on one dataset
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── CITATION.cff
└── LICENSE
```

The package is the original script split into modules **without changing
any function, class or parameter**. Each function's code is identical to the
one in `reference/xai_meta_qml_levelAB.py`, and the defaults in
`xqmids/config.py` are the settings used in the paper, so a run of the
package and a run of the original script give the same results.

## Installation

```bash
git clone https://github.com/youssefboulkhiout/explainable-quantum-meta-ids.git
cd explainable-quantum-meta-ids

# with conda (includes Spyder)
conda env create -f environment.yml
conda activate xqmids

# or with pip
pip install -r requirements.txt
```

## Data

Download the three datasets and save them as `data/ASEADOS_SDN_IoT.csv`,
`data/RT-IoT2022.csv` and `data/UNSW-NB15.csv`. Sources and label columns are
listed in [data/README.md](data/README.md).

## Running

### From Spyder

1. Open `run_in_spyder.py` in Spyder.
2. Edit the `OPTIONS` block if needed (leave the defaults to reproduce the paper).
3. Press **F5**.

The launcher switches to the repository folder, so `data/` and `outputs/`
are found wherever Spyder's working directory is. Figures are saved to disk
(not shown in the Plots pane), as in the original script.

### From a terminal

```bash
python -m xqmids                                   # all datasets, paper configuration
python -m xqmids --datasets RT-IoT2022 UNSW-NB15   # a subset of datasets
python -m xqmids --quick                           # fast smoke test
python -m xqmids --level B                         # only Level B quantum models
python -m xqmids --no-quantum                      # classical models only
```

| Option | Meaning | Paper value |
|---|---|---|
| `--datasets` | datasets to run | all three |
| `--level {A,B,AB}` | quantum extension(s) | `AB` |
| `--episodes N` | meta-training episodes of the classical meta-learners | 500 |
| `--max-rows N` | stratified row cap per dataset | 200000 |
| `--qubits q` | number of qubits / quantum features | 4 |
| `--eval-shots N` | finite-shot quantum evaluation | exact (not set) |
| `--embed-dim d` | Level B embedding dimension | 16 |
| `--output DIR` | output folder | `outputs` |
| `--quick` | reduced smoke-test configuration | not used |
| `--no-quantum` | disable all quantum models | not used |

Every dataset is seeded independently (`SEED = 42`), so running a subset of
datasets gives the same per-dataset results as running all three.

## Outputs

For each dataset, `outputs/<dataset>/` contains `model_metrics.csv`,
`fewshot_zeroday_metrics.csv`, `zeroday_class_report.csv`,
`zeroday_detection_metrics.csv`, `xai_metrics.csv`, `timing.csv`, loss
histories, circuit drawings, and a `figures/` folder (confusion matrices,
ROC curves, losses, SHAP/LIME plots, zero-day figures). The top-level
`outputs/` folder holds the cross-dataset files `all_model_metrics.csv`,
`all_fewshot_zeroday.csv`, `all_zeroday_class_report.csv`,
`all_zeroday_detection.csv`, `all_xai_metrics.csv` and
`comparison_figures/`. The tables and figures of the paper are built from
these five `all_*.csv` files.

## Reproducibility notes

- Results are reproducible for a given machine and library versions. Small
  differences can appear across hardware, thread counts, or versions of
  PyTorch, PennyLane, LightGBM and SHAP; record your versions with
  `pip freeze > versions.txt`.
- Quantum circuits are simulated on the CPU. PyTorch models use a CUDA GPU
  when one is available, otherwise the CPU. The full configuration is much
  slower than `--quick`, mainly because of the quantum simulations and the
  SHAP/LIME audit.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The
classical meta-learning benchmark that this work extends is:

> Y. Boulkhiout, L. Goudjil, S. Balbal, K. Nasri, "Explainable meta-learning
> for few-shot and zero-day intrusion detection in IoT networks",
> *Internet of Things*, 102144, 2026.

## License

MIT, see [LICENSE](LICENSE).
