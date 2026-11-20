"""Render simulated bid requests as OpenRTB 2.6 JSON.

The simulator keeps requests in a table for speed. The Go bidder reads
JSON as an exchange would send it, so this module turns rows back into
requests in each exchange's own dialect.
"""
from __future__ import annotations

import json

import pandas as pd

FORMAT = {
    "rewarded_video": {"video": {"mimes": ["video/mp4"], "w": 1080,
                                 "h": 1920, "protocols": [2, 3, 7]}},
    "interstitial": {"video": {"mimes": ["video/mp4"], "w": 1080,
                               "h": 1920, "protocols": [2, 3, 7]},
                     "banner": {"w": 320, "h": 480}},
    "banner": {"banner": {"w": 320, "h": 50}},
    "native": {"native": {"request": "{}", "ver": "1.2"}},
}


def to_openrtb(row) -> dict:
    """One bid request as an OpenRTB 2.6 dict."""
    imp = {"id": "1", "instl": int(row.instl), "bidfloorcur": "USD",
           **FORMAT[row.placement]}
    if row.bidfloor > 0:
        imp["bidfloor"] = float(row.bidfloor)
    if row.rwdd:
        imp["rwdd"] = 1
    if row.ext_rewarded:
        imp["ext"] = {"is_rewarded_inventory": 1}
    device = {"os": row.os, "osv": row.osv, "make": row.make,
              "connectiontype": int(row.connectiontype),
              "geo": {"country": row.country}}
    if row.model:
        device["model"] = row.model
    if row.ifa:
        device["ifa"] = row.ifa
        device["lmt"] = int(row.ifa.startswith("00000000-"))
    return {
        "id": row.request_id,
        "tmax": int(row.tmax),
        "at": int(row.at),
        "imp": [imp],
        "app": {"bundle": row.bundle},
        "device": device,
        "regs": {"coppa": 0},
    }


def write_jsonl(requests: pd.DataFrame, path: str) -> None:
    with open(path, "w") as f:
        for row in requests.itertuples():
            f.write(json.dumps(to_openrtb(row)) + "\n")
