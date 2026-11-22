"""Hashed one-hot tokens with crosses (Chapter 9).

Every rule here has a twin in the Go bidder (code/dsp/internal/model), and
the model artifact carries the spec, so the two never drift apart. The
golden test in each language checks the same tokens and hashes.
"""
from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field

import mmh3

HASH_FN = "murmur3_32"


@dataclass
class Spec:
    strings: list = field(default_factory=lambda: [
        # from the request
        "exchange", "publisher_app", "placement", "country", "os",
        "os_major", "make", "model", "connection_type",
        # from the campaign and creative being scored
        "campaign", "adv_app", "goal", "creative", "creative_format",
        "creative_length", "creative_orientation", "creative_style"])
    numeric_buckets: dict = field(default_factory=lambda: {
        "bidfloor": [0.25, 0.5, 1, 2, 4, 8],
        "hour_of_day": [3, 6, 9, 12, 15, 18, 21],
        "device_known": [1],
        "device_requests": [2, 5, 10, 20, 50, 100],
        "seconds_since_first_seen": [3600, 21600, 86400, 259200, 604800],
        "seconds_since_last_seen": [60, 600, 3600, 21600, 86400],
        "our_imps_today": [1, 2, 3, 5, 10],
        "seconds_since_our_last_imp": [60, 600, 3600, 21600],
    })
    sets: dict = field(default_factory=lambda: {"apps_seen": 30})
    crosses: list = field(default_factory=lambda: [
        ["placement", "campaign"],
        ["publisher_app", "campaign"],
        ["country", "campaign"],
        ["placement", "creative_format"],
        ["model", "publisher_app"],
    ])
    num_buckets: int = 2 ** 24
    seed: int = 0

    def to_json(self) -> dict:
        d = asdict(self)
        seed = d.pop("seed")
        d["hash"] = {"fn": HASH_FN, "seed": seed}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Spec":
        d = dict(d)
        h = d.pop("hash")
        if h["fn"] != HASH_FN:
            raise ValueError(f"unsupported hash {h['fn']!r}")
        return cls(seed=h["seed"], **d)

    def tokens(self, fv: dict, ad: dict) -> list[str]:
        """Tokens for one FeatureVector and ad, in the bidder's order."""
        strings = fv.get("strings") or {}
        numeric = fv.get("numeric") or {}
        sets = fv.get("sets") or {}

        def val(k):
            v = ad.get(k) or strings.get(k)
            return v if v else None

        toks = []
        for k in self.strings:
            v = val(k)
            toks.append(f"{k}_{v if v is not None else 'null'}")
        for k in sorted(self.numeric_buckets):
            if k not in numeric:
                toks.append(f"{k}_missing")
                continue
            i = bisect.bisect_right(self.numeric_buckets[k], numeric[k])
            toks.append(f"{k}_b{i}")
        for k in sorted(self.sets):
            for m in sorted(sets.get(k) or [])[: self.sets[k]]:
                toks.append(f"{k}_{m}")
        for a, b in self.crosses:
            va, vb = val(a), val(b)
            if va is not None and vb is not None:
                toks.append(f"{a}_{va}*{b}_{vb}")
        return toks

    def bucket(self, tok: str) -> int:
        return mmh3.hash(tok, self.seed, signed=False) % self.num_buckets

    def buckets(self, toks: list[str]) -> list[int]:
        """Distinct buckets, sorted. Colliding tokens count once."""
        return sorted({self.bucket(t) for t in toks})

    def label(self, tok: str) -> str:
        """A readable name for a token, for the console's readout."""
        parts = []
        for t in tok.split("*"):
            for k, edges in self.numeric_buckets.items():
                pre = k + "_b"
                if t.startswith(pre) and t[len(pre):].isdigit():
                    i = int(t[len(pre):])
                    lo = edges[i - 1] if i > 0 else None
                    hi = edges[i] if i < len(edges) else None
                    rng = (f"< {_num(hi)}" if lo is None else
                           f">= {_num(lo)}" if hi is None else
                           f"{_num(lo)} to {_num(hi)}")
                    t = f"{k} {rng}"
                    break
            parts.append(t)
        return " × ".join(parts)


def _num(x) -> str:
    return f"{x:g}"
