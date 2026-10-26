"""Stages 3Q/5C: Level-A quantum models (QSVC, VQC, QNN, HQCNN) and classical controls (SVM-q, MLP-q).

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import time
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ALL_MODELS, BASE_MODELS, META_MODELS, QCONTROL_MODELS, QUANTUM_B_MODELS, QUANTUM_MODELS, SEED
from .utils import log, sanitize_proba, savefig, set_seed

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
