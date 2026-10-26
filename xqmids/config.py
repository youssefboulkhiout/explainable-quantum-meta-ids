"""Global configuration: seed, device, CONFIG, dataset paths, model lists.

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import torch

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
