"""Stage 5D: Level-B hybrid quantum prototypical networks (QSVC-PN, VQC-PN, QNN-PN, HQCNN-PN, MLP-PN).

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import os
import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SEED
from .utils import _smooth, log, sanitize_proba, savefig, set_seed
from .quantum_a import _zz_feature_map, emulate_shots, qml, stratified_subsample

# =============================================================================
# 5D. QUANTUM MACHINE LEARNING -- LEVEL B
#     HYBRID QUANTUM PROTOTYPICAL NETWORKS (few-shot / zero-day)
# =============================================================================
# Level B turns each Level-A quantum model into the EMBEDDING NETWORK of a
# prototypical network that is META-TRAINED episodically (N-way K-shot), so
# the quantum circuit itself is optimised for few-shot generalisation:
#
#   x (q features) --encoding--> |psi(x)> --PQC U(theta)--> z = f_theta(x)
#   support set S_k  ->  prototype c_k = (1/K) sum_{i in S_k} z_i
#   query x          ->  p(y=k|x) = softmax_k( -tau * ||z_x - c_k||^2 )
#
#   QSVC-PN  : TRAINABLE QUANTUM-KERNEL prototypical network. The embedding is
#              the quantum state itself (|psi(x)><psi(x)| in Hilbert space):
#              ZZ feature map with learnable per-feature bandwidths, with a
#              variational layer between the two data-encoding repetitions
#              (trainable embedding kernel, Hubregtsen et al. 2022). The
#              prototype distance is the kernel-mean distance
#              ||phi(x) - mu_k||^2 = 1 - 2 mean_i K(x,x_i) + mean_ij K(x_i,x_j)
#              -- the Level-B, meta-learned counterpart of the Level-A QSVC.
#   VQC-PN   : fully quantum embedding. Encoding + StronglyEntanglingLayers;
#              z = expectations of d Pauli observables (single-qubit X/Y/Z and
#              neighbouring ZZ/XX correlators). No classical parameters.
#   QNN-PN   : data re-uploading circuit; z = Linear(<X>,<Y>,<Z> per qubit)->d.
#   HQCNN-PN : dressed circuit: classical pre-net -> PQC -> classical
#              post-net -> z in R^d (d = 8-32).
#   MLP-PN   : CLASSICAL control -- the same prototypical network with an MLP
#              embedding on the SAME q features and the SAME episodes, isolating
#              the effect of the quantum embedding.
#
# Like the classical meta-learners, Level-B models are meta-trained on the
# known classes only (meta_pool, zero-day excluded). At test time they need
# NO gradient steps: new or withheld attack families are handled just by
# computing prototypes from a few labelled examples (few-shot), and
# open-set zero-day detection uses MSP / entropy / margin over known
# prototypes, exactly as for every other model.
# =============================================================================
QUANTUM_B_KINDS = {"QSVC-PN": "QK", "VQC-PN": "VQC", "QNN-PN": "QNN",
                   "HQCNN-PN": "HQCNN", "MLP-PN": "MLP"}


def _pauli_pool(n):
    """Observable pool for the pure-quantum (VQC-PN) embedding: X, Y, Z on
    every qubit, then neighbouring ZZ and XX correlators."""
    pool = ([qml.PauliZ(i) for i in range(n)] +
            [qml.PauliX(i) for i in range(n)] +
            [qml.PauliY(i) for i in range(n)])
    pool += [qml.PauliZ(i) @ qml.PauliZ(i + 1) for i in range(n - 1)]
    pool += [qml.PauliX(i) @ qml.PauliX(i + 1) for i in range(n - 1)]
    return pool


def _encode(x, wires, encoding):
    """Classical-to-quantum encoding (pipeline step 3): angle (RY) or
    amplitude (q features padded to 2^q amplitudes, L2-normalised)."""
    if encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=wires, pad_with=0.0, normalize=True)
    else:
        qml.AngleEmbedding(x, wires=wires, rotation="Y")


class QProtoEmbed(nn.Module):
    """Embedding network f_theta of a Level-B prototypical network.
    kind: QK (quantum-kernel state embedding), VQC, QNN, HQCNN or MLP.
    embed(x) returns (B, 2^q) complex statevectors for QK, else (B, d) reals."""

    def __init__(self, kind, n, bcfg):
        super().__init__()
        self.kind, self.n = kind, n
        d, L = int(bcfg["embed_dim"]), int(bcfg["layers"])
        enc = bcfg.get("encoding", "angle")
        wires = list(range(n))
        # learnable inverse temperature of the distance softmax (all kinds,
        # incl. the classical control, so the comparison is like-for-like)
        self.log_tau = nn.Parameter(torch.tensor(np.log(bcfg["tau_init"]),
                                                 dtype=torch.float64))
        if kind == "MLP":
            self.net = nn.Sequential(nn.Linear(n, 64), nn.ReLU(),
                                     nn.Linear(64, d)).double()
            self.d = d
            return
        dev = qml.device(bcfg["device"], wires=n)

        if kind == "QK":
            reps = int(bcfg["fmap_reps"])
            self.scale = nn.Parameter(torch.full(
                (n,), float(bcfg["kernel_bandwidth_init"]), dtype=torch.float64))
            wshape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n)
            self.qw = nn.Parameter(torch.rand((max(reps - 1, 1),) + tuple(wshape),
                                              dtype=torch.float64) * 0.2)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(x, s, w):
                for r in range(reps):
                    _zz_feature_map(x * s, wires, 1)          # data layer
                    if r < reps - 1:                          # trainable layer
                        qml.StronglyEntanglingLayers(w[r], wires=wires)
                return qml.state()
            self.circuit, self.d = circuit, 2 ** n
            return

        wshape = qml.StronglyEntanglingLayers.shape(n_layers=L, n_wires=n)
        self.qw = nn.Parameter(torch.rand(wshape, dtype=torch.float64) * 2 * np.pi)
        if kind == "VQC":
            pool = _pauli_pool(n)
            if d > len(pool):
                log(f"    [Level B] VQC-PN: embed_dim {d} > {len(pool)} "
                    f"observables available with q={n}; using {len(pool)}")
            obs = pool[:d]
            self.d = len(obs)

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def circuit(x, w):
                _encode(x, wires, enc)
                qml.StronglyEntanglingLayers(w, wires=wires)
                return [qml.expval(o) for o in obs]
        else:
            obs = ([qml.PauliX(i) for i in wires] + [qml.PauliY(i) for i in wires]
                   + [qml.PauliZ(i) for i in wires])
            if kind == "QNN":
                # re-uploading needs a re-applicable encoding -> always angle
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def circuit(x, w):
                    for l in range(L):
                        qml.AngleEmbedding(x, wires=wires, rotation="Y")
                        qml.StronglyEntanglingLayers(w[l:l + 1], wires=wires)
                    return [qml.expval(o) for o in obs]
                self.post = nn.Linear(3 * n, d).double()
            else:                                             # HQCNN
                @qml.qnode(dev, interface="torch", diff_method="backprop")
                def circuit(x, w):
                    if enc != "amplitude":
                        for i in wires:
                            qml.Hadamard(wires=i)
                    _encode(x, wires, enc)
                    qml.StronglyEntanglingLayers(w, wires=wires)
                    return [qml.expval(o) for o in obs]
                self.pre = nn.Sequential(nn.Linear(n, n), nn.Tanh()).double()
                self.post = nn.Sequential(nn.Linear(3 * n, 32), nn.ReLU(),
                                          nn.Linear(32, d)).double()
            self.d = d
        self.circuit = circuit

    # -- raw quantum statistics (where finite-shot noise is injected) -------
    def _quantum(self, x):
        if self.kind == "QK":
            return self.circuit(x, self.scale, self.qw)
        if self.kind == "HQCNN":
            x = self.pre(x) * (np.pi / 2) + np.pi / 2       # back into [0, pi]
        out = self.circuit(x, self.qw)
        return torch.stack(out, dim=-1).reshape(x.shape[0], -1).to(torch.float64)

    def embed(self, x, shots=None, rng=None):
        """(B, q) angles -> embedding. With `shots`, expectation values are
        replaced by binomial finite-shot estimates (QK: handled in kernel)."""
        if self.kind == "MLP":
            return self.net(x)
        qo = self._quantum(x)
        if self.kind == "QK":
            return qo
        if shots:
            qo = torch.as_tensor(emulate_shots(qo.detach().numpy(), "expval",
                                               shots, rng))
        return self.post(qo) if self.kind in ("QNN", "HQCNN") else qo


def _kernel(SA, SB):
    """Fidelity kernel |<a|b>|^2 between two batches of statevectors (torch)."""
    return (SA.conj() @ SB.T).abs() ** 2


def proto_distances(net, z_q, z_s, ys, n_cls, shots=None, rng=None):
    """Squared distance of every query to every class prototype.
    QK: kernel-mean distance in Hilbert space; others: squared Euclidean
    distance to the mean embedding c_k."""
    if net.kind == "QK":
        Kqs, Kss = _kernel(z_q, z_s), _kernel(z_s, z_s)
        if shots:                                     # finite-shot kernel entries
            Kqs = torch.as_tensor(emulate_shots(Kqs.numpy(), "QSVC", shots, rng))
        d = []
        for k in range(n_cls):
            m = ys == k
            d.append(1.0 - 2.0 * Kqs[:, m].mean(1) + Kss[m][:, m].mean())
        return torch.stack(d, dim=1)
    protos = torch.stack([z_s[ys == k].mean(0) for k in range(n_cls)])
    return torch.cdist(z_q, protos) ** 2


def sample_episode_idx(y, pool, n_way, k_shot, q_query, rng):
    """Numpy N-way K-shot episode: support/query row indices + LOCAL labels."""
    classes = rng.choice(pool, size=min(n_way, len(pool)), replace=False)
    si, sl, qi, ql = [], [], [], []
    for j, c in enumerate(classes):
        idx = np.where(y == c)[0]
        take = rng.choice(idx, size=k_shot + q_query,
                          replace=len(idx) < k_shot + q_query)
        si.append(take[:k_shot]); sl += [j] * k_shot
        qi.append(take[k_shot:]); ql += [j] * q_query
    return (np.concatenate(si), np.array(sl), np.concatenate(qi),
            np.array(ql), classes)


def meta_train_qproto(name, Xa_tr, y_tr, class_pool, n, bcfg):
    """Episodic prototypical meta-training of one Level-B model."""
    set_seed()
    rng = np.random.RandomState(SEED)
    net = QProtoEmbed(QUANTUM_B_KINDS[name], n, bcfg)
    opt = torch.optim.Adam(net.parameters(), lr=bcfg["lr"])
    n_way = min(bcfg["n_way"], len(class_pool))
    X = torch.as_tensor(Xa_tr, dtype=torch.float64)
    history = []
    t0 = time.time()
    for _ in tqdm(range(bcfg["episodes"]), desc=f"\t\t\t[{name}]",
                  unit="episode", leave=False):
        opt.zero_grad()
        losses = []
        for _ in range(bcfg["meta_batch"]):
            si, sl, qi, ql, _ = sample_episode_idx(
                y_tr, class_pool, n_way, bcfg["k_shot"], bcfg["q_query"], rng)
            z = net.embed(X[np.concatenate([si, qi])])
            zs, zq = z[:len(si)], z[len(si):]
            dist = proto_distances(net, zq, zs, torch.as_tensor(sl), n_way)
            loss = F.cross_entropy(-net.log_tau.exp() * dist,
                                   torch.as_tensor(ql))
            (loss / bcfg["meta_batch"]).backward()
            losses.append(loss.item())
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        history.append(float(np.mean(losses)))
    net.eval()
    return net, time.time() - t0, history


class QProtoWrapper:
    """Prototype classifier built from a meta-trained Level-B embedding and a
    labelled support set (adapt_shot rows per class). Same interface as
    QuantumWrapper: predict_proba (30 features), predict_proba_q (q features,
    used by SHAP/LIME) and predict_proba_eval (finite shots if configured).
    No gradient steps are taken here -- only prototypes are computed."""

    def __init__(self, name, net, reducer, Xs, ys, n_space, bcfg):
        self.name, self.net, self.reducer = name, net, reducer
        self.n_space, self.bcfg = n_space, bcfg
        self.classes = np.unique(ys)
        self.classes_ = np.arange(n_space)
        loc = np.searchsorted(self.classes, ys)
        Xa = reducer.angles(reducer.reduce(Xs))
        with torch.no_grad():
            self.zs = net.embed(torch.as_tensor(Xa, dtype=torch.float64))
        self.ys = torch.as_tensor(loc)

    def _local(self, Xa, shots=None):
        rng = np.random.default_rng(SEED)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xa), 2048):
                xb = torch.as_tensor(Xa[i:i + 2048], dtype=torch.float64)
                zq = self.net.embed(xb, shots=shots, rng=rng)
                d = proto_distances(self.net, zq, self.zs, self.ys,
                                    len(self.classes), shots=shots, rng=rng)
                out.append(F.softmax(-self.net.log_tau.exp() * d, 1).numpy())
        return np.vstack(out)

    def predict_proba_q(self, Xq, shots=None):
        p = self._local(self.reducer.angles(np.atleast_2d(Xq)), shots)
        full = np.zeros((p.shape[0], self.n_space))
        full[:, self.classes] = p
        return sanitize_proba(full)

    def predict_proba(self, X):
        return self.predict_proba_q(self.reducer.reduce(np.atleast_2d(X)))

    def predict_proba_eval(self, X):
        return self.predict_proba_q(self.reducer.reduce(np.atleast_2d(X)),
                                    shots=self.bcfg.get("eval_shots"))

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def support_set(X, y, per_class, seed=SEED):
    """adapt_shot labelled rows per class (the few-shot support set)."""
    idx = stratified_subsample(y, per_class, np.random.RandomState(seed))
    return X[idx], y[idx]


def qproto_episodic_eval(net, Xa_te, y_te, pool, bcfg, tag):
    """Episodic N-way K-shot test evaluation (same protocol as the classical
    meta-learners): prototypes from K support shots of TEST classes, scored on
    the episode's queries. Returns mean and std over episodes."""
    rng = np.random.RandomState(SEED + 7)
    pool = [c for c in pool if np.sum(y_te == c) >= bcfg["k_shot"] + 2]
    if len(pool) < 2:
        return {}
    n_way = min(bcfg["n_way"], len(pool))
    X = torch.as_tensor(Xa_te, dtype=torch.float64)
    accs, f1s = [], []
    with torch.no_grad():
        for _ in range(bcfg["eval_episodes"]):
            si, sl, qi, ql, _ = sample_episode_idx(
                y_te, np.array(pool), n_way, bcfg["k_shot"], bcfg["q_query"], rng)
            z = net.embed(X[np.concatenate([si, qi])])
            d = proto_distances(net, z[len(si):], z[:len(si)],
                                torch.as_tensor(sl), n_way)
            pr = d.argmin(1).numpy()
            accs.append(accuracy_score(ql, pr))
            f1s.append(f1_score(ql, pr, average="macro", zero_division=0))
    return {f"{tag}_episodic_accuracy": float(np.mean(accs)),
            f"{tag}_episodic_accuracy_std": float(np.std(accs)),
            f"{tag}_episodic_f1_macro": float(np.mean(f1s)),
            f"{tag}_episodic_f1_macro_std": float(np.std(f1s)),
            f"{tag}_episodes": bcfg["eval_episodes"], f"{tag}_n_way": n_way}


def qproto_n_params(net):
    """(quantum-circuit parameters, total trainable parameters)."""
    total = sum(p.numel() for p in net.parameters())
    circ = 0
    if hasattr(net, "qw"):
        circ += net.qw.numel()
    if hasattr(net, "scale"):
        circ += net.scale.numel()
    return int(circ), int(total)


# ---- Level-B figures / artefacts ----------------------------------------------
def save_levelB_circuits(nets, reducer, path):
    """Text drawings of the Level-B embedding circuits."""
    x = torch.as_tensor(np.linspace(0.3, 2.8, reducer.q))
    lines = [f"Level-B quantum embedding circuits (q = {reducer.q}, "
             f"features: {', '.join(reducer.names)})", ""]
    for name, net in nets.items():
        if net.kind == "MLP":
            continue
        try:
            args = ((x, net.scale.detach(), net.qw.detach()) if net.kind == "QK"
                    else (x, net.qw.detach()))
            lines += [f"--- {name} ---",
                      qml.draw(net.circuit, decimals=2, max_length=160,
                               show_matrices=False)(*args), ""]
        except Exception as e:
            lines.append(f"--- {name}: drawing failed ({e}) ---")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_levelB_losses(histories, ds_name, path):
    """Smoothed episodic meta-training loss of every Level-B model."""
    if not histories:
        return
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name, h in histories.items():
        s = _smooth(h, frac=0.05)
        ax.plot(np.arange(len(s)), s, lw=1.5, label=name,
                color=None if name != "MLP-PN" else "#555555",
                ls="--" if name == "MLP-PN" else "-")
    ax.set_xlabel("Meta-training episode")
    ax.set_ylabel("Prototypical query loss (smoothed)")
    ax.set_title(f"{ds_name} -- Level-B hybrid quantum ProtoNet training")
    ax.legend(fontsize=8)
    savefig(fig, path)


def plot_levelB_embeddings(wrappers, X_te, y_te, class_names, zero_day,
                           ds_name, figdir, n_per_class=60):
    """2-D PCA of test embeddings + class prototypes for each Level-B model;
    withheld zero-day families are drawn as crosses (they were never seen in
    meta-training -- whether they form their own cluster is the point)."""
    from sklearn.decomposition import PCA
    rng = np.random.RandomState(SEED)
    idx = stratified_subsample(y_te, n_per_class, rng)
    Xs, ys = X_te[idx], y_te[idx]
    for name, w in wrappers.items():
        try:
            Xa = w.reducer.angles(w.reducer.reduce(Xs))
            with torch.no_grad():
                z = w.net.embed(torch.as_tensor(Xa, dtype=torch.float64))
                zp = w.zs
            if w.net.kind == "QK":       # complex states -> real feature vector
                to_r = lambda s: torch.cat([s.real, s.imag], 1)
                z, zp = to_r(z), to_r(zp)
            z, zp = z.numpy(), zp.numpy()
            protos = np.stack([zp[w.ys.numpy() == k].mean(0)
                               for k in range(len(w.classes))])
            pca = PCA(2, random_state=SEED).fit(np.vstack([z, protos]))
            Z, P = pca.transform(z), pca.transform(protos)
            fig, ax = plt.subplots(figsize=(7, 6))
            cmap = plt.get_cmap("tab20")
            for j, c in enumerate(np.unique(ys)):
                m = ys == c
                ax.scatter(Z[m, 0], Z[m, 1], s=12, alpha=0.6,
                           marker="x" if c in zero_day else "o",
                           color=cmap(j % 20),
                           label=class_names[c][:16] +
                           (" (zero-day)" if c in zero_day else ""))
            ax.scatter(P[:, 0], P[:, 1], s=160, marker="*", c="k",
                       edgecolors="w", label="prototypes")
            ax.set_title(f"{ds_name} -- {name} embedding space (PCA)")
            ax.legend(fontsize=6, ncol=2, loc="best")
            savefig(fig, os.path.join(figdir, f"levelB_embedding_{name}.png"))
        except Exception as e:
            log(f"    embedding plot failed for {name}: {e}")


# ---- bootstrap confidence intervals (error bars of pipeline step 8) -----------
def bootstrap_ci(fn, *arrays, B=200, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI of fn(*arrays) over resampled rows."""
    if not B:
        return np.nan, np.nan
    rng = np.random.RandomState(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(B):
        i = rng.randint(0, n, n)
        try:
            vals.append(fn(*[a[i] for a in arrays]))
        except Exception:
            continue
    if not vals:
        return np.nan, np.nan
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


FAMILIES = OrderedDict([
    ("Classical", ["LightGBM", "CNN"]),
    ("Classical meta-learning", ["Reptile", "MAML", "FOMAML", "Meta-SGD",
                                 "ANIL", "BOIL", "ProtoNet"]),
    ("Level A (QSVC/VQC/QNN/HQCNN)", ["QSVC", "VQC", "QNN", "HQCNN"]),
    ("Level B (hybrid quantum ProtoNet)", ["QSVC-PN", "VQC-PN", "QNN-PN",
                                           "HQCNN-PN"]),
])
FAMILY_COLORS = ["#3070b3", "#d62728", "#ff7f0e", "#2ca02c"]


def plot_level_comparison(perf, zdet, title, path):
    """Pipeline step 8: Classical vs Level A vs Level B. For each metric the
    best model of each family is shown. One dataset -> error bars are 95%
    bootstrap CIs; several datasets -> mean +- std over datasets of each
    family's best model. The winning model's name is printed on each bar."""
    if perf is None or perf.empty:
        return
    df = perf.copy()
    if zdet is not None and not zdet.empty:
        z = zdet[zdet["Score"] == "MSP"][["Dataset", "Model", "AUROC",
                                          "AUROC_lo", "AUROC_hi"]]
        df = df.merge(z, on=["Dataset", "Model"], how="left")
    metrics = [("Accuracy", "Accuracy"), ("F1", "Macro-F1"),
               ("AUROC", "Zero-day AUROC")]
    metrics = [(m, l) for m, l in metrics if m in df.columns]
    datasets = list(dict.fromkeys(df["Dataset"]))
    fams = [(f, [m for m in ms if m in set(df["Model"])])
            for f, ms in FAMILIES.items()]
    fams = [(f, ms) for f, ms in fams if ms]
    fig, ax = plt.subplots(figsize=(3.2 + 2.6 * len(metrics), 5))
    width = 0.8 / max(len(fams), 1)
    for j, (fam, ms) in enumerate(fams):
        vals, lo, hi, labels = [], [], [], []
        for met, _ in metrics:
            sub = df[df["Model"].isin(ms)]
            best = sub.loc[sub.groupby("Dataset")[met].idxmax().dropna()]
            if best.empty:
                vals.append(np.nan); lo.append(0); hi.append(0); labels.append("")
                continue
            v = best[met].mean()
            if len(datasets) == 1 and f"{met}_lo" in best.columns:
                el = v - best[f"{met}_lo"].iloc[0]
                eh = best[f"{met}_hi"].iloc[0] - v
            else:
                el = eh = best[met].std(ddof=0) if len(best) > 1 else 0.0
            vals.append(v); lo.append(np.nan_to_num(el)); hi.append(np.nan_to_num(eh))
            labels.append(best["Model"].mode().iloc[0])
        x = np.arange(len(metrics)) + (j - (len(fams) - 1) / 2) * width
        ax.bar(x, np.nan_to_num(vals), width, yerr=[lo, hi], capsize=3,
               color=FAMILY_COLORS[j % len(FAMILY_COLORS)], label=fam)
        for xi, v, lab in zip(x, vals, labels):
            if lab and np.isfinite(v):
                ax.text(xi, 0.02, lab, rotation=90, ha="center", va="bottom",
                        fontsize=7, color="white")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([l for _, l in metrics])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Metric value (best model per family)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=2, frameon=False)
    savefig(fig, path)
