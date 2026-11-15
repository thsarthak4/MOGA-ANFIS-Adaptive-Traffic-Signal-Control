"""Sugeno ANFIS (zero-order) with 2 inputs (queue length, arrival rate), 5 Gaussian MFs each, 25 rules.
Parameter vector (45 genes): [cQL(5), sQL(5), cV(5), sV(5), consequents(25)]"""
import numpy as np

QL_MAX, V_MAX = 150.0, 20.0
LEVELS = {"VL": 0.0, "L": 0.25, "M": 0.5, "H": 0.75, "VH": 1.0}
# Rule table from the paper (Table 1). Rows = vehicle arrival V (VS..VL), cols = queue length QL (VS..VL)
RULES = [["VL", "VL", "VL", "L", "L"],
         ["VL", "L",  "L",  "M", "M"],
         ["VL", "L",  "M",  "M", "H"],
         ["L",  "M",  "M",  "H", "VH"],
         ["L",  "M",  "H",  "VH", "VH"]]
RULE_OUT = np.array([[LEVELS[x] for x in r] for r in RULES])    # (5 V, 5 QL)


def default_params():
    cq = np.array([0, 30, 60, 90, 120.]); sq = np.full(5, 13.0)
    cv = np.array([0, 4, 8, 12, 16.]);    sv = np.full(5, 2.0)
    return np.concatenate([cq, sq, cv, sv, RULE_OUT.flatten()])


def repair(p):
    p = np.array(p, float).copy()
    p[0:5] = np.sort(p[0:5]); p[10:15] = np.sort(p[10:15])
    p[5:10] = np.maximum(p[5:10], 1.0); p[15:20] = np.maximum(p[15:20], 0.3)
    p[20:] = np.clip(p[20:], 0, 1)
    return p


class ANFIS:
    def __init__(self, params=None):
        self.p = default_params() if params is None else repair(params)

    def norm_firing(self, ql, v):
        p = self.p
        mq = np.exp(-(ql[:, None] - p[0:5]) ** 2 / (2 * p[5:10] ** 2))    # layer 1
        mv = np.exp(-(v[:, None] - p[10:15]) ** 2 / (2 * p[15:20] ** 2))
        w = (mv[:, :, None] * mq[:, None, :]).reshape(len(ql), 25)        # layer 2 (product AND)
        return w / (w.sum(1, keepdims=True) + 1e-12)                      # layer 3 (normalise)

    def predict(self, ql, v):                                             # layers 4+5
        return np.clip(self.norm_firing(ql, v) @ self.p[20:], 0, 1)


def make_dataset(n=3750, seed=0):
    """Synthetic dataset generated from the rule table (as in the paper: 3750 samples)."""
    rng = np.random.default_rng(seed)
    ql = rng.uniform(0, QL_MAX, n); v = rng.uniform(0, V_MAX, n)
    b = default_params()
    bq = np.digitize(ql, (b[0:4] + b[1:5]) / 2)
    bv = np.digitize(v, (b[10:14] + b[11:15]) / 2)
    return np.c_[ql, v], RULE_OUT[bv, bq]


def rmse(m, X, y):
    return float(np.sqrt(np.mean((m.predict(X[:, 0], X[:, 1]) - y) ** 2)))


def pretrain(epochs=80, seed=0):
    """Hybrid learning (Jang 1993): least squares for consequents + gradient step for MF params."""
    X, y = make_dataset(seed=seed)
    n = len(y); i1, i2 = int(.7 * n), int(.9 * n)
    Xtr, ytr, Xck, yck, Xte, yte = X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:]
    m = ANFIS(); lr = 1.0
    scale = np.array([3] * 5 + [1] * 5 + [0.4] * 5 + [0.2] * 5)
    hist = []
    for ep in range(epochs):
        W = m.norm_firing(Xtr[:, 0], Xtr[:, 1])
        c, *_ = np.linalg.lstsq(W, ytr, rcond=None)
        m.p[20:] = np.clip(c, 0, 1)
        cur = rmse(m, Xtr, ytr)
        grad = np.zeros(20)
        for i in range(20):                                   # finite-difference gradient
            old = m.p[i]; m.p[i] = old + 1e-3
            grad[i] = (rmse(m, Xtr, ytr) - cur) / 1e-3
            m.p[i] = old
        trial = m.p.copy()
        trial[:20] -= lr * scale * grad / (np.abs(grad).max() + 1e-12)
        cand = ANFIS(trial)
        if rmse(cand, Xtr, ytr) < cur: m = cand
        else: lr *= 0.5
        hist.append((rmse(m, Xtr, ytr), rmse(m, Xck, yck)))
    W = m.norm_firing(Xtr[:, 0], Xtr[:, 1])                   # final LSE
    m.p[20:] = np.clip(np.linalg.lstsq(W, ytr, rcond=None)[0], 0, 1)
    info = {"train": rmse(m, Xtr, ytr), "check": rmse(m, Xck, yck), "test": rmse(m, Xte, yte), "hist": hist}
    return m, info