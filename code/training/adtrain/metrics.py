"""Offline metrics for a click model (Chapters 9 and 11).

Log loss and normalized entropy, ROC and PR curves with their areas,
calibration by decile, and segment breakdowns. Plain numpy, so the
numbers are easy to check by hand.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def normalized_entropy(y, p) -> float:
    """Log loss divided by that of always predicting the average rate."""
    base = float(np.mean(y))
    if base <= 0 or base >= 1:
        return float("nan")
    return log_loss(y, p) / log_loss(y, np.full(len(y), base))


def roc(y, p, max_points: int = 200):
    """ROC curve points (fpr, tpr) and the AUC."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return [], float("nan")
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    # one point per distinct score, so ties are handled correctly
    last = np.r_[np.diff(ps) != 0, True]
    tp = np.cumsum(ys)[last]
    fp = np.cumsum(1 - ys)[last]
    tpr = np.r_[0, tp / pos]
    fpr = np.r_[0, fp / neg]
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))
    return _thin(fpr, tpr, max_points), auc


def pr(y, p, max_points: int = 200):
    """Precision-recall points (recall, precision) and average precision."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    pos = y.sum()
    if pos == 0:
        return [], float("nan")
    order = np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    last = np.r_[np.diff(ps) != 0, True]
    tp = np.cumsum(ys)[last]
    k = np.arange(1, len(ys) + 1)[last]
    precision = tp / k
    recall = tp / pos
    ap = float(np.sum(np.diff(np.r_[0, recall]) * precision))
    return _thin(recall, precision, max_points), ap


def calibration(y, p, bins: int = 10, true_p=None):
    """Mean prediction and observed rate in equal-count bins of the
    prediction, lowest first. With the simulator's true rates, also the
    mean true rate, which separates model error from click noise."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    out = []
    for idx in np.array_split(order, bins):
        if len(idx) == 0:
            continue
        row = {"n": int(len(idx)), "pred": float(p[idx].mean()),
               "actual": float(y[idx].mean())}
        if true_p is not None:
            t = np.asarray(true_p, float)[idx]
            t = t[~np.isnan(t)]
            if len(t):
                row["true"] = float(t.mean())
        out.append(row)
    return out


def summary(y, p) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    _, auc = roc(y, p)
    _, ap = pr(y, p)
    clicks = int(y.sum())
    return {
        "n": int(len(y)), "clicks": clicks,
        "ctr": float(y.mean()) if len(y) else float("nan"),
        "mean_pred": float(p.mean()) if len(p) else float("nan"),
        "log_loss": log_loss(y, p) if len(y) else float("nan"),
        "ne": normalized_entropy(y, p) if len(y) else float("nan"),
        "auc": auc, "pr_auc": ap,
        # predicted clicks over actual clicks: 1.0 is calibrated
        "calibration_ratio": float(p.sum() / clicks) if clicks else float("nan"),
    }


def segments(keys, y, p, logged, min_n: int = 30):
    """Metrics by segment, largest first."""
    keys = np.asarray(keys, dtype=object)
    out = []
    for k in np.unique(keys):
        m = keys == k
        if m.sum() < min_n:
            continue
        s = summary(np.asarray(y)[m], np.asarray(p)[m])
        s["mean_logged"] = float(np.asarray(logged)[m].mean())
        s["segment"] = str(k)
        out.append(s)
    return sorted(out, key=lambda r: -r["n"])


def cosine(a: dict, b: dict) -> float:
    """Cosine similarity of two sparse weight vectors, by bucket."""
    keys = set(a) | set(b)
    if not keys:
        return float("nan")
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return float("nan")
    return float(va @ vb / (na * nb))


def _thin(x, y, max_points):
    n = len(x)
    if n > max_points:
        idx = np.unique(np.linspace(0, n - 1, max_points).round().astype(int))
        x, y = x[idx], y[idx]
    return [[round(float(a), 5), round(float(b), 5)] for a, b in zip(x, y)]
