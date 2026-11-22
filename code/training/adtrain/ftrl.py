"""FTRL-Proximal logistic regression on hashed buckets (Chapter 9).

Each bucket keeps two numbers, z and n, and its weight is computed from
them when needed. Only the buckets present in an example are touched, so
memory and time grow with the number of distinct buckets seen, never with
the size of the hash space. The bias is a bucket of its own, with no L1,
so it's never zeroed out.
"""
from __future__ import annotations

import math

BIAS = -1  # the bias's key; real buckets are >= 0


class FTRL:
    def __init__(self, alpha: float = 0.1, beta: float = 1.0,
                 l1: float = 1.0, l2: float = 1.0):
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        self.z: dict[int, float] = {}
        self.n: dict[int, float] = {}

    def start_bias(self, rate: float) -> None:
        """Start the bias at the log-odds of the base click rate, instead
        of at 0, which predicts 50%. From 50%, the first examples push the
        weights of common tokens far below zero, and one pass never fully
        undoes it: the model ends up predicting too low everywhere."""
        rate = min(max(rate, 1e-6), 1 - 1e-6)
        logit = math.log(rate / (1 - rate))
        # with n = 0, weight = -z / (beta / alpha + l2)
        self.z[BIAS] = -logit * (self.beta / self.alpha + self.l2)
        self.n[BIAS] = 0.0

    def weight(self, i: int) -> float:
        z = self.z.get(i, 0.0)
        l1 = 0.0 if i == BIAS else self.l1
        if abs(z) <= l1:
            return 0.0
        n = self.n.get(i, 0.0)
        return -(z - math.copysign(l1, z)) / (
            (self.beta + math.sqrt(n)) / self.alpha + self.l2)

    def predict(self, buckets: list[int]) -> float:
        s = self.weight(BIAS) + sum(self.weight(i) for i in buckets)
        return 1.0 / (1.0 + math.exp(-max(min(s, 35.0), -35.0)))

    def update(self, buckets: list[int], y: int) -> float:
        """One online step. Returns the prediction made before it."""
        keys = [BIAS, *buckets]
        w = {i: self.weight(i) for i in keys}
        s = sum(w.values())
        p = 1.0 / (1.0 + math.exp(-max(min(s, 35.0), -35.0)))
        g = p - y  # the gradient for every active bucket, as x_i = 1
        g2 = g * g
        for i in keys:
            n = self.n.get(i, 0.0)
            sigma = (math.sqrt(n + g2) - math.sqrt(n)) / self.alpha
            self.z[i] = self.z.get(i, 0.0) + g - sigma * w[i]
            self.n[i] = n + g2
        return p

    def recenter_bias(self, rows, iters: int = 20) -> float:
        """Shift the bias so the mean prediction on `rows` equals their
        click rate, as it would for a fully converged logistic regression.
        Online updates leave the bias wherever the last examples pushed
        it, so this makes the final model's overall rate stable from run
        to run. Returns the shift, in log-odds."""
        bias = self.weight(BIAS)
        rest = [sum(self.weight(i) for i in b) for b, _ in rows]
        target = sum(y for _, y in rows)
        shift = 0.0
        for _ in range(iters):  # Newton steps on one variable
            ps = [1 / (1 + math.exp(-max(min(bias + shift + r, 35), -35)))
                  for r in rest]
            g = sum(ps) - target
            h = sum(q * (1 - q) for q in ps)
            if h <= 0 or abs(g) < 1e-6:
                break
            shift -= g / h
        self.bias_shift = shift
        return shift

    bias_shift = 0.0

    def weights(self) -> tuple[float, dict[int, float]]:
        """The bias, and the nonzero weights by bucket."""
        out = {}
        for i in self.z:
            if i == BIAS:
                continue
            w = self.weight(i)
            if w != 0.0:
                out[i] = w
        return self.weight(BIAS) + self.bias_shift, out
