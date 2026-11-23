"""Live mode against a stand-in bidder, so these tests need no Go."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import pytest

from adsim.config import preset
from adsim.live import TIME_HEADER, HttpPolicy
from adsim.run import simulate


class Bidder(BaseHTTPRequestHandler):
    """Bids $1 on banners, sleeps past tmax on native, no-bids the rest."""
    protocol_version = "HTTP/1.1"  # keep connections open, like the Go bidder
    seen = []
    notices = []
    tokens = set()
    books = []
    invoices = []
    bid_all = False
    book = None
    slow_native = True

    def do_PUT(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/invoices":
            Bidder.invoices = body
        else:
            Bidder.book = body
            Bidder.books.append(body)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    postback_times = []

    def do_GET(self):
        Bidder.notices.append(self.path)
        if self.path.startswith("/postback"):
            Bidder.postback_times.append(int(self.headers[TIME_HEADER]))
        if self.path.startswith("/postback"):
            Bidder.tokens.add(self.headers.get("X-MMP-Token"))
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Bidder.seen.append((self.path, int(self.headers[TIME_HEADER])))
        imp = req["imp"][0]
        if "native" in imp and Bidder.slow_native:
            time.sleep(0.3)  # longer than tmax = 250 ms
        if ("banner" in imp and "video" not in imp) or Bidder.bid_all:
            q = f"rid={req['id']}"
            body = json.dumps({"id": req["id"], "seatbid": [{"bid": [{
                "id": "1", "impid": "1", "price": 1.0, "cid": "1",
                "crid": "1",
                "nurl": f"http://x/notice/win?{q}&price=${{AUCTION_PRICE}}",
                "burl": f"http://x/notice/billing?{q}&price=${{AUCTION_PRICE}}",
                "lurl": f"http://x/notice/loss?{q}&reason=${{AUCTION_LOSS}}",
                "adm": json.dumps({"native": {"link": {
                    "url": "https://apps.example/1",
                    "clicktrackers": [f"http://x/click?{q}"]}}}),
            }]}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def bidder_url():
    ThreadingHTTPServer.request_queue_size = 64
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Bidder)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_http_policy_answers(bidder_url):
    cfg = preset("crawl", "tiny", seed=3, days=1)
    from adsim.world import make_world
    from adsim.history import make_history
    from adsim.requests import make_requests
    from adsim.run import PUBLIC_REQUEST_COLS
    world = make_world(cfg)
    _, installed = make_history(world)
    req = make_requests(world, installed, 0).head(400)
    pol = HttpPolicy(bidder_url, workers=8)
    dec = pol.decide(req[PUBLIC_REQUEST_COLS], {})
    pol.close()
    by = dict(zip(req.request_id, req.placement))
    for rid, action, reason in zip(dec.request_id, dec.action, dec.reason):
        p = by[rid]
        if p == "banner":
            assert action == "bid"
        elif p == "native":
            assert (action, reason) == ("nobid", "timeout")
        else:
            assert (action, reason) == ("nobid", "nobid")
    # Each exchange has its own path, and the simulated time is sent.
    paths = {p for p, _ in Bidder.seen}
    assert paths <= {"/openrtb/alpha_exchange", "/openrtb/beta_exchange"}
    assert sorted(t for _, t in Bidder.seen)[:5] == sorted(req.ts_ms)[:5]


def test_listen_only_run_has_well_formed_logs(bidder_url):
    class Listen:
        def decide(self, req, state):
            import pandas as pd
            return pd.DataFrame({"request_id": req.request_id,
                                 "ts_ms": req.ts_ms, "action": "nobid",
                                 "reason": "listen"})
    import dataclasses
    cfg = dataclasses.replace(preset("crawl", "tiny", seed=3, days=1),
                              requests_per_device=4.0)
    res = simulate(cfg, policy=Listen())
    assert len(res.logs["bid_log"]) == len(res.logs["bid_requests"])
    assert (res.logs["bid_log"].action == "nobid").all()
    assert len(res.logs["impressions"]) == 0


def test_notices_follow_the_auctions(bidder_url):
    """A live run pushes the campaign book, then sends a win or loss
    notice for every bid (on exchanges that send losses), and a billing
    notice for every shown impression."""
    Bidder.notices.clear()
    Bidder.slow_native = False
    import dataclasses
    cfg = dataclasses.replace(preset("crawl", "tiny", seed=5, days=1),
                              requests_per_device=4.0)  # a small market
    pol = HttpPolicy(bidder_url, workers=8, admin_url=bidder_url,
                     notice_retry=0)
    res = simulate(cfg, policy=pol)
    pol.close()
    assert Bidder.book and len(Bidder.book["campaigns"]) == len(res.logs["campaigns"])
    assert "quality" not in Bidder.book["creatives"][0]  # truth stays hidden
    kinds = [p.split("?")[0] for p in Bidder.notices]
    notes = res.logs["notifications"]
    notes = notes.drop_duplicates("event_id")
    for kind in ("win", "loss", "billing"):
        assert kinds.count(f"/notice/{kind}") == (notes.type == kind).sum()
    # Every click on a shown ad calls its click tracker once.
    assert kinds.count("/click") == len(res.logs["clicks"]) > 0
    # The MMP sends every postback, with its token.
    assert kinds.count("/postback") == len(res.logs["postbacks"])
    assert Bidder.tokens <= {"adsim-mmp"}
    # Macros are filled in: prices on wins and billing, codes on losses.
    assert all("AUCTION" not in p for p in Bidder.notices)
    assert any("reason=102" in p for p in Bidder.notices)


def test_scenarios_and_invoices(bidder_url):
    """wrong-target pushes a second campaign book with one CPC target 100x;
    lost-billing drops beta_exchange's rewarded video billing from 12:00;
    the exchanges' invoices cover every shown impression either way."""
    import dataclasses
    from urllib.parse import parse_qs, urlsplit
    Bidder.notices.clear()
    Bidder.books.clear()
    Bidder.slow_native, Bidder.bid_all = False, True
    cfg = dataclasses.replace(preset("crawl", "tiny", seed=5, days=1),
                              requests_per_device=4.0)
    pol = HttpPolicy(bidder_url, workers=8, admin_url=bidder_url,
                     notice_retry=0, scenarios=("wrong-target", "lost-billing"))
    try:
        res = simulate(cfg, policy=pol)
    finally:
        pol.close()
        Bidder.bid_all = False
    first, second = Bidder.books[0]["campaigns"], Bidder.books[1]["campaigns"]
    changed = [(a["target"], b["target"]) for a, b in zip(first, second)
               if a["target"] != b["target"]]
    assert len(changed) == 1 and abs(changed[0][1] - 100 * changed[0][0]) < 1e-9
    # billing notices for rewarded video on beta stop at 12:00
    req = res.logs["bid_requests"].set_index("request_id")
    late, before = 0, 0
    for path in Bidder.notices:
        if not path.startswith("/notice/billing"):
            continue
        rid = parse_qs(urlsplit(path).query)["rid"][0]
        r = req.loc[rid]
        if r.exchange == "beta_exchange" and r.placement == "rewarded_video":
            if r.ts_ms >= 12 * 3_600_000:
                late += 1
            else:
                before += 1
    assert before > 0 and late <= 2  # only impressions bid just before 12:00
    inv = res.logs["invoices"]
    assert inv.impressions.sum() == len(res.logs["impressions"])
    assert Bidder.invoices and len(Bidder.invoices) == len(inv)
    assert ((inv.exchange == "beta_exchange")
            & (inv.placement == "rewarded_video")).any()


def test_postbacks_wait_for_their_time(bidder_url, tmp_path):
    """Across runs, the MMP never sends a postback before its time: the
    ones due after a run ends wait in a file and go out in the next run."""
    import dataclasses
    Bidder.slow_native = False
    pending = tmp_path / "state" / "pending_postbacks.parquet"
    sent = []
    for start in (0, 1):
        Bidder.postback_times.clear()
        cfg = dataclasses.replace(preset("crawl", "tiny", seed=5, days=1),
                                  requests_per_device=4.0, start_day=start)
        pol = HttpPolicy(bidder_url, workers=8, admin_url=bidder_url,
                         notice_retry=0, pending_path=str(pending))
        res = simulate(cfg, policy=pol)
        pol.close()
        end = (start + 1) * 86_400_000
        assert Bidder.postback_times and max(Bidder.postback_times) <= end
        pb = res.logs["postbacks"]
        if start == 0:
            assert pending.exists()
            later = pd.read_parquet(pending)
            assert (later.received_ts_ms > end).all()
            assert len(later) == (pb.received_ts_ms > end).sum()
        sent.append(len(Bidder.postback_times))
    # the second run sent its own postbacks and some of the first run's
    assert sent[1] > 0

