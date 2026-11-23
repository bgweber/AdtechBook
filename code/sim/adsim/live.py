"""Live mode: the simulated exchanges send bid requests to a real bidder.

`HttpPolicy` has the same interface as the Python policies, so `simulate`
can run against either. For each request it does what an exchange does:
render the request as OpenRTB JSON in that exchange's dialect, POST it to
the bidder at /openrtb/{exchange}, and wait at most `tmax` milliseconds.
A response that arrives late counts as a no-bid, so a slow bidder loses
auctions just as it would in production.

The simulated time goes in the X-Adsim-Time header, so the bidder's
budgets, day boundaries, and logs follow the simulated clock.

Around the auctions, it plays the rest of the exchange's part: it pushes
the campaign book to the bidder's admin API when a run starts (`setup`),
and calls the notice URLs in each bid, `nurl` on a win, `lurl` on a loss,
and `burl` when the ad is shown and billed, with OpenRTB macros such as
${AUCTION_PRICE} filled in (`notify`). Like real exchanges, it retries a
small share of notices, so the bidder has to deduplicate them.

When a run ends and installs are resolved, the simulated MMP sends its
install and event postbacks to the bidder's postback endpoint, with a
shared token in X-MMP-Token and its own field names (`postbacks`).

When a shown ad is clicked, it calls the click trackers in the ad markup,
as the ad SDK on the device would. The markup is a small native-style
JSON object, {"native": {"link": {"url": ..., "clicktrackers": [...]}}},
whatever the placement, which keeps the simulated SDK simple.
"""
from __future__ import annotations

import http.client
import json
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode, urlparse, urlsplit

import numpy as np
import pandas as pd

from .openrtb import to_openrtb

TIME_HEADER = "X-Adsim-Time"

# Failure scenarios for the operations exercises (Chapter 12). Each starts
# some hours into the first simulated day of the run.
SCENARIOS = {
    # the campaign system pushes a CPC target entered 100 times too high
    "wrong-target": "at 06:00, the first CPC campaign's target becomes 100x",
    # an exchange stops sending billing notices for one ad format
    "lost-billing": "from 12:00, beta_exchange sends no billing notices for rewarded video",
}
WRONG_TARGET_HOUR = 6
LOST_BILLING_HOUR = 12
LOST_BILLING_PLACEMENT = "rewarded_video"


class HttpPolicy:
    """Send each request to a bidder over HTTP and record its answer."""

    def __init__(self, url: str = "http://localhost:8080", workers: int = 32,
                 sim_epoch_ms: int = 0, admin_url: str | None = None,
                 notice_retry: float = 0.005, seed: int = 0,
                 mmp_token: str = "adsim-mmp", scenarios: tuple = (),
                 pending_path: str | None = None):
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port or 80
        self.admin = admin_url or f"http://{self.host}:8081"
        self.notice_retry = notice_retry
        self.rng = np.random.default_rng([seed, 200])
        self.urls = {}  # request ID -> (nurl, burl, lurl, ts) from our bids
        self.clicktrackers = {}  # request ID -> click tracker URLs
        self.notices_sent = 0
        self.clicks_sent = 0
        self.postbacks_sent = 0
        self.mmp_token = mmp_token
        # Postbacks the MMP will send after this run ends wait in a file
        # and go out during the next run, when their time comes, so the
        # bidder never hears about an install before it happens.
        self.pending_path = pending_path
        self.pending = _load_pending(pending_path)
        self.scenarios = set(scenarios)
        unknown = self.scenarios - set(SCENARIOS)
        if unknown:
            raise ValueError(f"unknown scenarios {sorted(unknown)}; "
                             f"choose from {sorted(SCENARIOS)}")
        self.book = None  # the campaign book pushed at setup
        self.first_ms = None  # simulated time of the first request
        self.done = set()  # scenarios already triggered
        self.info = {}  # request ID -> (exchange, placement), for our bids
        self.workers = workers
        self.epoch = sim_epoch_ms
        self.local = threading.local()
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.version = ""
        self.sent = 0
        self.wall_s = 0.0
        self.wait_for_bidder()

    def wait_for_bidder(self, timeout_s: float = 60.0) -> None:
        """Wait until the bidder accepts connections, as when it's still
        starting in another container."""
        import socket
        deadline = time.time() + timeout_s
        while True:
            try:
                socket.create_connection((self.host, self.port), 1).close()
                return
            except OSError:
                if time.time() > deadline:
                    raise RuntimeError(
                        f"no bidder at {self.host}:{self.port}")
                time.sleep(0.5)

    def _conn(self):
        c = getattr(self.local, "conn", None)
        if c is None:
            c = http.client.HTTPConnection(self.host, self.port, timeout=5)
            self.local.conn = c
        return c

    def _send(self, row) -> tuple:
        body = json.dumps(to_openrtb(row)).encode()
        headers = {"Content-Type": "application/json",
                   TIME_HEADER: str(int(row.ts_ms) + self.epoch)}
        for attempt in range(2):  # one retry on a dropped keep-alive
            conn = self._conn()
            t0 = time.perf_counter()
            try:
                conn.request("POST", f"/openrtb/{row.exchange}", body, headers)
                resp = conn.getresponse()
                data = resp.read()
                ms = (time.perf_counter() - t0) * 1000
                self.version = resp.getheader("X-Bidder-Version", self.version)
                return resp.status, data, ms
            except (http.client.HTTPException, OSError):
                conn.close()
                self.local.conn = None
                if attempt:
                    return None, b"", (time.perf_counter() - t0) * 1000
        return None, b"", 0.0

    def _hours_in(self, now_ms: int) -> float:
        if self.first_ms is None:
            self.first_ms = now_ms - now_ms % 86_400_000
        return (now_ms - self.first_ms) / 3_600_000

    def _scenarios(self, now_ms: int) -> None:
        h = self._hours_in(now_ms)
        if ("wrong-target" in self.scenarios and "wrong-target" not in self.done
                and h >= WRONG_TARGET_HOUR and self.book is not None):
            self.done.add("wrong-target")
            camps, crs = self.book
            camps = camps.copy()
            i = camps.index[camps.goal == "CPC"][0]
            old = camps.at[i, "target"]
            camps.at[i, "target"] = old * 100
            self._put_book(camps, crs)
            print(f"scenario wrong-target: campaign {camps.at[i, 'campaign_id']} "
                  f"target {old:g} -> {old * 100:g} at {_clock(now_ms)}", flush=True)

    def _lost_billing(self, rid: str, t_ms: int) -> bool:
        if "lost-billing" not in self.scenarios:
            return False
        ex, pl = self.info.get(rid, ("", ""))
        return (ex == "beta_exchange" and pl == LOST_BILLING_PLACEMENT
                and self._hours_in(t_ms) >= LOST_BILLING_HOUR)

    def decide(self, req: pd.DataFrame, state: dict) -> pd.DataFrame:
        self._send_pending(int(req.ts_ms.min()))
        if self.scenarios:
            self._scenarios(int(state.get("now_ms", req.ts_ms.min())))
        else:
            self._hours_in(int(req.ts_ms.min()))
        t0 = time.perf_counter()
        rows = list(req.itertuples())
        results = list(self.pool.map(self._send, rows))
        self.wall_s += time.perf_counter() - t0
        self.sent += len(rows)
        n = len(rows)
        action = np.full(n, "nobid", dtype=object)
        reason = np.full(n, "", dtype=object)
        bid = np.full(n, np.nan)
        camp = np.full(n, np.nan)
        crea = np.full(n, np.nan)
        lat = np.array([r[2] for r in results])
        tmax = req.tmax.to_numpy()
        for i, (status, data, ms) in enumerate(results):
            if status is None:
                reason[i] = "bidder_error"
            elif ms > tmax[i]:
                reason[i] = "timeout"
            elif status == 204:
                reason[i] = "nobid"
            elif status == 200:
                try:
                    b = json.loads(data)["seatbid"][0]["bid"][0]
                    action[i], bid[i] = "bid", float(b["price"])
                    camp[i] = float(b.get("cid", "nan"))
                    crea[i] = float(b.get("crid", "nan"))
                    self.urls[rows[i].request_id] = (
                        b.get("nurl"), b.get("burl"), b.get("lurl"),
                        int(rows[i].ts_ms))
                    self.info[rows[i].request_id] = (rows[i].exchange,
                                                     rows[i].placement)
                    ct = _clicktrackers(b.get("adm"))
                    if ct:
                        self.clicktrackers[rows[i].request_id] = ct
                except (ValueError, KeyError, IndexError, TypeError):
                    reason[i] = "bad_response"
            else:
                reason[i] = f"http_{status}"
        return pd.DataFrame({
            "request_id": req.request_id.to_numpy(),
            "ts_ms": req.ts_ms.to_numpy(), "action": action,
            "reason": reason, "campaign_id": camp, "creative_id": crea,
            "bid": bid, "latency_ms": lat,
            "policy_version": f"http:{self.version}"})

    def next_day(self) -> int:
        """The day after the latest one in the bidder's ledger, or 0."""
        u = urlsplit(self.admin)
        try:
            conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
            conn.request("GET", "/api/ledger")
            resp = conn.getresponse()
            body = resp.read()
            conn.close()
            return int(json.loads(body)["next_day"]) if resp.status == 200 else 0
        except (OSError, ValueError, KeyError, http.client.HTTPException):
            return 0

    def setup(self, campaigns: pd.DataFrame, creatives: pd.DataFrame) -> None:
        """Push the campaign book to the bidder's admin API."""
        self.book = (campaigns, creatives)
        self._put_book(campaigns, creatives)

    def _put_book(self, campaigns: pd.DataFrame, creatives: pd.DataFrame) -> None:
        body = json.dumps({
            "campaigns": json.loads(campaigns.to_json(orient="records")),
            "creatives": json.loads(creatives.to_json(orient="records")),
        }).encode()
        self._admin("PUT", "/api/campaigns", body, "the campaign book")

    def invoices(self, inv: pd.DataFrame) -> None:
        """Send each exchange's invoice, as exchanges do after the fact."""
        body = inv.to_json(orient="records").encode()
        self._admin("PUT", "/api/invoices", body, "the invoices")

    def _admin(self, method: str, path: str, body: bytes, what: str) -> None:
        u = urlsplit(self.admin)
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=10)
        conn.request(method, path, body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status != 200:
            raise RuntimeError(f"bidder rejected {what}: HTTP {resp.status}")

    def notify(self, results: pd.DataFrame, imps: pd.DataFrame) -> None:
        """Send win, loss, and billing notices for one chunk of auctions.

        `results` has one row per bid with the auction outcome; `imps` has
        one row per shown, billed impression. Win and loss notices carry
        the auction's time; billing notices the time the ad was shown.
        """
        jobs = []  # (url, simulated time)
        for r in results.itertuples():
            urls = self.urls.get(r.request_id)
            if urls is None:
                continue
            nurl, _, lurl, t = urls
            mtw = "" if pd.isna(r.reported_mbtw) else f"{r.reported_mbtw:.2f}"
            if r.win:
                jobs.append((self._fill(nurl, r.price, mtw), t))
            else:
                if r.sends_loss:
                    code = "100" if r.below_floor else "102"  # OpenRTB codes
                    jobs.append((self._fill(lurl, None, mtw, code), t))
                del self.urls[r.request_id]
        clicks = []
        has_clicks = "clicked" in imps.columns
        for r in imps.itertuples():
            urls = self.urls.pop(r.request_id, None)
            if urls is not None and not self._lost_billing(r.request_id, int(r.imp_ts_ms)):
                jobs.append((self._fill(urls[1], r.price, ""), int(r.imp_ts_ms)))
            self.info.pop(r.request_id, None)
            ct = self.clicktrackers.pop(r.request_id, None)
            if ct and has_clicks and r.clicked:
                clicks += [(u, int(r.click_ts_ms)) for u in ct]
        for rid in set(results.request_id[results.win]) - set(imps.request_id):
            self.urls.pop(rid, None)  # won but never shown: never billed
            self.clicktrackers.pop(rid, None)
        for rid in results.request_id[~results.win]:
            self.clicktrackers.pop(rid, None)
            self.info.pop(rid, None)
        retries = [j for j in jobs if self.rng.random() < self.notice_retry]
        jobs += retries  # exchanges retry some notices
        # SDKs retry some clicks too
        clicks += [c for c in clicks if self.rng.random() < self.notice_retry]
        if jobs or clicks:
            list(self.pool.map(self._get, jobs + clicks))
            self.notices_sent += len(jobs)
            self.clicks_sent += len(clicks)

    @staticmethod
    def _fill(url: str, price, mtw: str, loss: str = "") -> str:
        p = "" if price is None or pd.isna(price) else f"{price:.4f}"
        return (url.replace("${AUCTION_PRICE}", p)
                   .replace("${AUCTION_MIN_TO_WIN}", mtw)
                   .replace("${AUCTION_LOSS}", loss))

    def postbacks(self, pb: pd.DataFrame, until_ms: int | None = None) -> None:
        """Send the MMP's postbacks, in the order the MMP sends them.

        With a `pending_path`, postbacks the MMP sends after `until_ms`, the
        end of the run, are kept for the next run rather than sent now.
        """
        if self.pending is not None and not self.pending.empty:
            pb = pd.concat([self.pending, pb], ignore_index=True)
            self.pending = None
        pb = pb.sort_values("received_ts_ms", kind="stable")
        if until_ms is not None and self.pending_path:
            later = pb[pb.received_ts_ms > until_ms]
            pb = pb[pb.received_ts_ms <= until_ms]
            self._save_pending(later)
        self._send_postbacks(pb)

    def _send_pending(self, now_ms: int) -> None:
        if self.pending is None or self.pending.empty:
            return
        due = self.pending[self.pending.received_ts_ms <= now_ms]
        if not due.empty:
            self.pending = self.pending[self.pending.received_ts_ms > now_ms]
            self._send_postbacks(due)

    def _save_pending(self, later: pd.DataFrame) -> None:
        self.pending = later.reset_index(drop=True)
        if not self.pending_path:
            return
        Path(self.pending_path).parent.mkdir(parents=True, exist_ok=True)
        later.reset_index(drop=True).to_parquet(self.pending_path, index=False)

    def _send_postbacks(self, pb: pd.DataFrame) -> None:
        if pb.empty:
            return
        cols = [c for c in pb.columns if c != "received_ts_ms"]
        jobs = []
        for r in pb.itertuples(index=False):
            d = r._asdict()
            q = {c: ("" if pd.isna(d[c]) else d[c]) for c in cols}
            jobs.append(("/postback?" + urlencode(q), int(d["received_ts_ms"]),
                         {"X-MMP-Token": self.mmp_token}))
        list(self.pool.map(self._get, jobs))
        self.postbacks_sent += len(jobs)

    def _get(self, job) -> None:
        url, t = job[0], job[1]
        extra = job[2] if len(job) > 2 else {}
        u = urlsplit(url)
        path = u.path + ("?" + u.query if u.query else "")
        for attempt in range(2):
            conn = self._conn()
            try:
                conn.request("GET", path, headers={TIME_HEADER: str(t + self.epoch), **extra})
                conn.getresponse().read()
                return
            except (http.client.HTTPException, OSError):
                conn.close()
                self.local.conn = None

    def close(self):
        self.pool.shutdown()

    @property
    def rate(self) -> float:
        """Requests per wall-clock second so far."""
        return self.sent / self.wall_s if self.wall_s else 0.0


def _clicktrackers(adm) -> list[str]:
    """Click tracker URLs from native-style ad markup, if any."""
    if not adm:
        return []
    try:
        link = json.loads(adm)["native"]["link"]
        return [u for u in link.get("clicktrackers", []) if isinstance(u, str)]
    except (ValueError, KeyError, TypeError):
        return []


def _clock(ms: int) -> str:
    m = (ms % 86_400_000) // 60_000
    return f"day {ms // 86_400_000}, {m // 60:02d}:{m % 60:02d}"


def _load_pending(path: str | None) -> pd.DataFrame | None:
    if path and Path(path).exists():
        return pd.read_parquet(path)
    return None

