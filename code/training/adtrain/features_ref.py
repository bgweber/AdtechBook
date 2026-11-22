"""A second implementation of the bidder's normalization and FeatureVector
(Chapters 6 and 8), for parity checks.

The Go bidder computes features at bid time and logs them. This module
recomputes them from the raw context the bidder logs for a sample of
bids: the request as the exchange sent it, and the device profile the
bidder read. If the two ever disagree, one of them changed, and a model
trained on one is being served the other.
"""
from __future__ import annotations

DAY_MS = 86_400_000
ZERO_IFA = "00000000-0000-0000-0000-000000000000"
ALPHA2_TO_3 = {"IN": "IND", "BR": "BRA", "ID": "IDN", "MX": "MEX",
               "US": "USA", "JP": "JPN", "DE": "DEU"}


def normalize(req: dict, exchange: str) -> dict:
    """The internal request, as code/dsp/internal/normalize builds it."""
    imp = (req.get("imp") or [{}])[0]
    ext = imp.get("ext") or {}
    rewarded = imp.get("rwdd") == 1 or (
        isinstance(ext, dict) and ext.get("is_rewarded_inventory") == 1)
    if imp.get("native") is not None:
        placement = "native"
    elif imp.get("video") is not None:
        placement = "rewarded_video" if rewarded else "interstitial"
    else:
        placement = "banner"
    app = req.get("app") or {}
    dev = req.get("device") or {}
    geo = dev.get("geo") or {}
    ifa = dev.get("ifa") or ""
    country = (geo.get("country") or "").strip().upper()
    if len(country) == 2:
        country = ALPHA2_TO_3.get(country, country)
    return {
        "exchange": exchange, "bundle": _clean_bundle(app.get("bundle") or ""),
        "placement": placement, "rewarded": rewarded,
        "instl": imp.get("instl") == 1, "floor": float(imp.get("bidfloor") or 0),
        "country": country, "os": dev.get("os") or "", "osv": dev.get("osv") or "",
        "make": dev.get("make") or "", "model": dev.get("model") or "",
        "conntype": int(dev.get("connectiontype") or 0),
        "ifa": ifa, "valid_ifa": ifa not in ("", ZERO_IFA),
    }


def _clean_bundle(b: str) -> str:
    i = b.find("$")
    if i >= 0:
        b = b[:i]
    return b.strip().lower()


def build(n: dict, p: dict, ts_ms: int) -> dict:
    """The FeatureVector, as code/dsp/internal/features builds it."""
    strings, numeric = {}, {}

    def s(k, v):
        v = (v or "").strip().lower()
        if v:
            strings[k] = v  # an absent key means missing

    s("exchange", n["exchange"])
    s("publisher_app", n["bundle"])
    s("placement", n["placement"])
    s("country", n["country"])
    s("os", n["os"])
    s("os_major", n["osv"].split(".", 1)[0])
    s("make", n["make"])
    s("model", n["model"])
    if n["conntype"] > 0:
        strings["connection_type"] = {2: "wifi", 1: "ethernet"}.get(
            n["conntype"], "cellular" if 3 <= n["conntype"] <= 7 else "other")
    numeric["bidfloor"] = max(n["floor"], 0.0)
    numeric["rewarded"] = float(n["rewarded"])
    numeric["interstitial"] = float(n["instl"])
    numeric["hour_of_day"] = float((ts_ms % DAY_MS) // 3_600_000)
    known = bool(p.get("known"))
    numeric["device_known"] = float(known)
    fv = {"schema_version": 1, "numeric": numeric, "strings": strings}
    if known:
        numeric["device_requests"] = float(p["requests"])
        numeric["seconds_since_first_seen"] = _secs(ts_ms - p["first_seen_ms"])
        numeric["seconds_since_last_seen"] = _secs(ts_ms - p["last_seen_ms"])
        numeric["our_imps_today"] = float(p.get("imps_today", 0))
        if p.get("last_imp_ms", 0) > 0:
            numeric["seconds_since_our_last_imp"] = _secs(ts_ms - p["last_imp_ms"])
        if p.get("apps"):
            fv["sets"] = {"apps_seen": list(p["apps"])}
        if p.get("recent"):
            fv["lists"] = {"recent_apps": {
                "values": [v["app"] for v in p["recent"]],
                "age_seconds": [int(_secs(ts_ms - v["ts_ms"])) for v in p["recent"]]}}
    return fv


def _secs(ms: int) -> float:
    return 0.0 if ms < 0 else float(ms // 1000)


def flatten(fv: dict) -> dict:
    """One value per feature name, for comparing two FeatureVectors."""
    out = {}
    for k, v in (fv.get("strings") or {}).items():
        out[k] = v
    for k, v in (fv.get("numeric") or {}).items():
        out[k] = float(v)
    for k, v in (fv.get("sets") or {}).items():
        out[k] = tuple(sorted(v))
    for k, v in (fv.get("lists") or {}).items():
        out[k] = (tuple(v.get("values") or []), tuple(v.get("age_seconds") or []))
    return out
