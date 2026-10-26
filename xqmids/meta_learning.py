"""Stage 5B: functional-MLP meta-learning framework (Reptile, MAML, FOMAML, Meta-SGD, ANIL, BOIL, ProtoNet).

Part of explainable-quantum-meta-ids. Code moved verbatim from the
single-file script xai_meta_qml_levelAB.py; logic and parameters unchanged.
"""


import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE, SEED
from .utils import sanitize_proba, set_seed

# =============================================================================
# 5B. META-LEARNING FRAMEWORK (functional MLP, 7 algorithms)
# =============================================================================
# Instead of an nn.Module, meta-parameters are plain tensors in an OrderedDict.
# This "functional" style lets each algorithm clone/perturb/partially-update
# the weights per-episode without fighting PyTorch's module/optimizer state.
BODY_KEYS = ["w1", "b1", "w2", "b2"]     # shared feature-extractor weights
HEAD_KEYS = ["head_w", "head_b"]         # final classification layer


def init_meta_params(in_dim, n_classes, hidden=(128, 64)):
    """Random-init a 2-hidden-layer MLP (body) + linear head as a flat dict
    of leaf tensors, so it can be adapted/cloned functionally per episode."""
    h1, h2 = hidden
    def lin(o, i):
        """Kaiming-uniform weight (o x i) + zero bias, both requiring grad."""
        w = torch.empty(o, i, device=DEVICE)
        nn.init.kaiming_uniform_(w, a=np.sqrt(5))
        b = torch.zeros(o, device=DEVICE)
        return w.requires_grad_(), b.requires_grad_()
    p = OrderedDict()
    p["w1"], p["b1"] = lin(h1, in_dim)
    p["w2"], p["b2"] = lin(h2, h1)
    p["head_w"], p["head_b"] = lin(n_classes, h2)
    return p


def body_forward(x, p):
    """Feature extractor only (used by ProtoNet prototypes / ANIL / BOIL)."""
    x = F.relu(F.linear(x, p["w1"], p["b1"]))
    x = F.relu(F.linear(x, p["w2"], p["b2"]))
    return x                                     # embedding


def full_forward(x, p):
    """Body + classification head -> class logits."""
    return F.linear(body_forward(x, p), p["head_w"], p["head_b"])


def clone_params(p, detach=True):
    """Deep-copy the parameter dict. detach=True starts a fresh autograd
    graph (used for each episode's inner-loop adaptation)."""
    return OrderedDict((k, (v.detach().clone().requires_grad_() if detach
                            else v) ) for k, v in p.items())


def sample_episode(X, y, class_pool, n_way, k_shot, q_query, rng):
    """N-way K-shot episode using ORIGINAL label ids (shared global head)."""
    classes = rng.choice(class_pool, size=min(n_way, len(class_pool)),
                         replace=False)
    sx, sy, qx, qy = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        # sample support (k_shot) + query (q_query) examples for this class;
        # allow sampling with replacement if the class doesn't have enough rows
        take = rng.choice(idx, size=min(len(idx), k_shot + q_query),
                          replace=len(idx) < k_shot + q_query)
        sx.append(X[take[:k_shot]]);        sy.append(np.full(k_shot, c))
        qx.append(X[take[k_shot:]]);        qy.append(np.full(len(take) - k_shot, c))
    to_t = lambda a, dt: torch.as_tensor(np.concatenate(a), dtype=dt, device=DEVICE)
    return (to_t(sx, torch.float32), to_t(sy, torch.long),
            to_t(qx, torch.float32), to_t(qy, torch.long))


def inner_adapt(params, sx, sy, update_keys, inner_lr, steps,
                second_order=False, lr_dict=None):
    """Gradient-based inner loop; returns adapted (fast) parameters.
    - update_keys: which parameter tensors get adapted (all/body-only/head-only)
      -> this is what distinguishes MAML (all) from ANIL (head) / BOIL (body).
    - second_order=True keeps the graph through the inner steps (true MAML),
      False detaches it each step (first-order approximation, e.g. FOMAML).
    - lr_dict, if given, supplies a PER-PARAMETER learned step size (Meta-SGD)
      instead of a single scalar inner_lr."""
    fast = OrderedDict(params)
    for _ in range(steps):
        loss = F.cross_entropy(full_forward(sx, fast), sy)
        grads = torch.autograd.grad(loss, [fast[k] for k in update_keys],
                                    create_graph=second_order)
        new = OrderedDict(fast)
        for k, g in zip(update_keys, grads):
            step = (lr_dict[k] * g) if lr_dict is not None else inner_lr * g
            new[k] = fast[k] - step
        fast = new
    return fast


def proto_logits(sx, sy, qx, params):
    """ProtoNet logits = negative squared Euclidean distance to prototypes.
    Prototypes are the per-class mean embedding of the support set; query
    points are classified by nearest prototype (metric-based, no gradient
    adaptation at inference time)."""
    emb_s, emb_q = body_forward(sx, params), body_forward(qx, params)
    classes = torch.unique(sy)
    protos = torch.stack([emb_s[sy == c].mean(0) for c in classes])
    d = torch.cdist(emb_q, protos) ** 2
    return -d, classes


def meta_train(algo, X_tr, y_tr, class_pool, in_dim, n_classes, mcfg):
    """Episodic meta-training for one algorithm; returns meta-parameters.

    All 7 algorithms share the same episodic sampling and the same MLP body,
    they only differ in (a) which parameters get inner-loop-adapted per task,
    (b) whether the inner loop keeps second-order gradients, and (c) how the
    outer/meta update is applied (Adam meta-gradient vs. Reptile's direct
    parameter interpolation)."""
    set_seed()
    rng = np.random.RandomState(SEED)
    params = init_meta_params(in_dim, n_classes, mcfg["hidden"])
    n_way = min(mcfg["n_way"], len(class_pool))
    ks, qq, steps, ilr = (mcfg["k_shot"], mcfg["q_query"],
                          mcfg["inner_steps"], mcfg["inner_lr"])

    # ---- per-algorithm choice of (a) adaptable params (b) 2nd-order grads ----
    if algo == "ANIL":
        upd_keys, second = HEAD_KEYS, False       # adapt head only, first-order
    elif algo == "BOIL":
        upd_keys, second = BODY_KEYS, False        # adapt body only, first-order
    elif algo == "MAML":
        upd_keys, second = list(params.keys()), True   # adapt everything, true 2nd-order
    elif algo in ("FOMAML", "Meta-SGD", "Reptile"):
        # all adapt every parameter; Meta-SGD additionally learns per-param LR
        upd_keys, second = list(params.keys()), (algo == "Meta-SGD")
    else:                                        # ProtoNet
        upd_keys, second = BODY_KEYS, False        # no gradient inner loop at all

    lr_dict = None
    trainables = list(params.values())
    if algo == "Meta-SGD":
        # learnable per-parameter inner learning rates (in addition to the
        # parameters themselves) -- this is what makes Meta-SGD "Meta-SGD"
        lr_dict = OrderedDict((k, torch.full_like(v, ilr).requires_grad_())
                              for k, v in params.items())
        trainables += list(lr_dict.values())

    if algo != "Reptile":
        # Reptile doesn't use an outer optimizer/meta-gradient at all; it
        # directly interpolates params toward the post-adaptation weights.
        opt = torch.optim.Adam(trainables, lr=mcfg["meta_lr"])
    
    loss = 0.0    
    history = []                       # per-episode training loss
    start_train = time.time()
    for ep in tqdm(range(mcfg["episodes"]), desc=f"\t\t\t[{algo}]", unit="episode", leave=False):
    #for ep in range(mcfg["episodes"]):
        if algo == "Reptile":                                     # ---- Reptile
            # linearly anneal the meta step size from reptile_eps -> 0
            eps = mcfg["reptile_eps"] * (1 - ep / mcfg["episodes"])
            ep_losses = []
            for _ in range(mcfg["meta_batch"]):
                sx, sy, qx, qy = sample_episode(X_tr, y_tr, class_pool,
                                                n_way, ks, qq, rng)
                # Reptile trains on support+query jointly (no separate query loss)
                tx = torch.cat([sx, qx]); ty = torch.cat([sy, qy])
                fast = clone_params(params)
                iopt = torch.optim.SGD(fast.values(), lr=mcfg["reptile_lr"])
                for _ in range(steps):              # k SGD steps on this task
                    iopt.zero_grad()
                    l_task = F.cross_entropy(full_forward(tx, fast), ty)
                    l_task.backward()
                    iopt.step()
                ep_losses.append(float(l_task))
                # meta-update: move params a small step toward the task-adapted
                # weights (Reptile's first-order approximation of MAML)
                with torch.no_grad():
                    for k in params:
                        params[k] += (eps / mcfg["meta_batch"]) * \
                                     (fast[k].detach() - params[k])
            history.append(float(np.mean(ep_losses)))
            continue

        opt.zero_grad()
        meta_loss = 0.0
        for _ in range(mcfg["meta_batch"]):         # average meta-gradient over a batch of tasks
            sx, sy, qx, qy = sample_episode(X_tr, y_tr, class_pool,
                                            n_way, ks, qq, rng)
            if algo == "ProtoNet":                                # ---- ProtoNet
                # metric-based: no inner-loop gradient steps, just compute
                # prototypes from the support set and classify the query set
                logits, classes = proto_logits(sx, sy, qx, params)
                y_map = torch.searchsorted(classes, qy)  # map global ids -> episode-local ids
                task_loss = F.cross_entropy(logits, y_map)
            else:                                                 # gradient-based
                if algo == "FOMAML":
                    # first-order MAML: adapt on the support set, then compute
                    # the query loss and its gradient w.r.t. the ADAPTED params,
                    # but manually copy that gradient onto the ORIGINAL params
                    # (skipping backprop through the inner-loop steps entirely)
                    fast = inner_adapt(params, sx, sy, upd_keys, ilr, steps,
                                       second_order=False)
                    fast = OrderedDict((k, v.detach().requires_grad_()
                                        if k in upd_keys else v)
                                       for k, v in fast.items())
                    task_loss = F.cross_entropy(full_forward(qx, fast), qy)
                    grads = torch.autograd.grad(task_loss,
                                                [fast[k] for k in upd_keys])
                    for k, g in zip(upd_keys, grads):   # first-order transport
                        g = g / mcfg["meta_batch"]
                        params[k].grad = (params[k].grad + g
                                          if params[k].grad is not None else g.clone())
                    meta_loss += task_loss.item()
                    continue
                # MAML (true 2nd-order), ANIL, BOIL, Meta-SGD all funnel through
                # here: adapt on support set (fast weights), evaluate query loss
                # on the fast weights, and let autograd backprop through the
                # whole inner loop into the ORIGINAL params (opt.step() below).
                fast = inner_adapt(params, sx, sy, upd_keys, ilr, steps,
                                   second_order=second, lr_dict=lr_dict)
                task_loss = F.cross_entropy(full_forward(qx, fast), qy)
            (task_loss / mcfg["meta_batch"]).backward()
            meta_loss += float(task_loss)
        # clip the meta-gradient to keep the outer update from exploding into
        # NaN weights (a common cause of non-finite softmax outputs later)
        torch.nn.utils.clip_grad_norm_(trainables, max_norm=5.0)
        opt.step()                                  # single meta-update per episode
        loss = meta_loss / mcfg['meta_batch']
        history.append(float(loss))
    
    training_time = time.time() - start_train
    #log(f"    [{algo}] meta-loss={loss:.4f}")
    return params, lr_dict, training_time, history


class MetaWrapper:
    """Globally-adapted meta-model exposing predict_proba on the full test set."""
    def __init__(self, algo, params, n_classes, protos=None, proto_classes=None):
        """Hold the (globally-adapted) meta-parameters; for ProtoNet also hold
        the class prototypes and their global class ids."""
        self.algo, self.params, self.classes_ = algo, params, np.arange(n_classes)
        self.protos, self.proto_classes = protos, proto_classes

    def _logits(self, xb):
        """Per-class logits: nearest-prototype distances for ProtoNet, else the
        MLP body+head forward pass. Absent ProtoNet classes get -1e9."""
        if self.algo == "ProtoNet":
            emb = body_forward(xb, self.params)
            d = torch.cdist(emb, self.protos) ** 2
            logits = torch.full((xb.shape[0], len(self.classes_)), -1e9,
                                device=DEVICE)
            logits[:, self.proto_classes] = -d
            return logits
        return full_forward(xb, self.params)

    def predict_proba(self, X):
        """Batched, no-grad softmax over the full label space, as (n, n_classes)."""
        X = np.asarray(X, dtype=np.float32)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 4096):      # chunk to bound GPU memory
                xb = torch.as_tensor(X[i:i + 4096]).to(DEVICE)
                outs.append(F.softmax(self._logits(xb), dim=1).cpu().numpy())
        return sanitize_proba(np.vstack(outs))

    def predict(self, X):
        """Hard class predictions = argmax of predict_proba."""
        return np.argmax(self.predict_proba(X), axis=1)


def global_adapt(algo, params, lr_dict, X_tr, y_tr, n_classes, mcfg):
    """Few-shot global adaptation: small support set over ALL train classes.

    After meta-training (which only ever sees n_way classes per episode),
    this step adapts the meta-learned initialization to the FULL label space
    using just `adapt_shot` examples per class, producing the model that is
    actually evaluated on the full test set (see run_dataset's evaluation
    stage)."""
    rng = np.random.RandomState(SEED)
    sx, sy = [], []
    for c in np.unique(y_tr):
        idx = np.where(y_tr == c)[0]
        take = rng.choice(idx, size=min(len(idx), mcfg["adapt_shot"]),
                          replace=False)
        sx.append(X_tr[take]); sy.append(np.full(len(take), c))
    sx = torch.as_tensor(np.concatenate(sx), dtype=torch.float32, device=DEVICE)
    sy = torch.as_tensor(np.concatenate(sy), dtype=torch.long, device=DEVICE)

    if algo == "ProtoNet":
        # no gradient steps: just recompute prototypes over ALL classes
        with torch.no_grad():
            emb = body_forward(sx, params)
            classes = torch.unique(sy)
            protos = torch.stack([emb[sy == c].mean(0) for c in classes])
        return MetaWrapper(algo, clone_params(params), n_classes,
                           protos=protos, proto_classes=classes.cpu().numpy())

    # gradient-based algorithms: fine-tune the relevant parameter subset with
    # plain SGD for `adapt_steps` steps on the global support set
    upd_keys = (HEAD_KEYS if algo == "ANIL"
                else BODY_KEYS if algo == "BOIL"
                else list(params.keys()))
    fast = clone_params(params)
    opt = torch.optim.SGD([fast[k] for k in upd_keys], lr=mcfg["inner_lr"])
    for _ in range(mcfg["adapt_steps"]):
        opt.zero_grad()
        F.cross_entropy(full_forward(sx, fast), sy).backward()
        opt.step()
    for v in fast.values():
        v.requires_grad_(False)             # freeze for inference
    return MetaWrapper(algo, fast, n_classes)


def episodic_eval(algo, params, lr_dict, X_te, y_te, class_pool, mcfg,
                  tag="fewshot"):
    """Episodic N-way K-shot evaluation (few-shot / zero-day accuracy & F1).

    Unlike global_adapt (adapt once on train, evaluate on the whole test
    set), this simulates the "true" few-shot setting: for each of
    `eval_episodes` episodes, sample a fresh support/query split from the
    TEST classes in `class_pool`, adapt from the meta-parameters using only
    that episode's support set, then score on its query set. `tag` is
    "fewshot" for meta_pool classes or "zeroday" for held-out classes."""
    rng = np.random.RandomState(SEED + 7)
    # only keep classes with enough test rows to form a support+query episode
    pool = [c for c in class_pool
            if np.sum(y_te == c) >= mcfg["k_shot"] + 2]
    if len(pool) < 2:
        return {}
    n_way = min(mcfg["n_way"], len(pool))
    accs, f1s = [], []
    for _ in range(mcfg["eval_episodes"]):
        sx, sy, qx, qy = sample_episode(X_te, y_te, np.array(pool), n_way,
                                        mcfg["k_shot"], mcfg["q_query"], rng)
        if algo == "ProtoNet":
            with torch.no_grad():
                logits, classes = proto_logits(sx, sy, qx, params)
                pred = classes[logits.argmax(1)]
        else:
            upd_keys = (HEAD_KEYS if algo == "ANIL"
                        else BODY_KEYS if algo == "BOIL"
                        else list(params.keys()))
            fast = clone_params(params)
            opt = torch.optim.SGD([fast[k] for k in upd_keys],
                                  lr=mcfg["inner_lr"])
            # extra adaptation steps (2x inner_steps) since this starts from
            # the meta-learned init with no prior global adaptation
            for _ in range(mcfg["inner_steps"] * 2):
                opt.zero_grad()
                F.cross_entropy(full_forward(sx, fast), sy).backward()
                opt.step()
            with torch.no_grad():
                pred = full_forward(qx, fast).argmax(1)
        yq, pr = qy.cpu().numpy(), pred.cpu().numpy()
        accs.append(accuracy_score(yq, pr))
        f1s.append(f1_score(yq, pr, average="macro", zero_division=0))
    return {f"{tag}_episodic_accuracy": float(np.mean(accs)),
            f"{tag}_episodic_f1_macro": float(np.mean(f1s)),
            f"{tag}_episodes": mcfg["eval_episodes"], f"{tag}_n_way": n_way}
