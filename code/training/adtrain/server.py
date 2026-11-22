"""The training service the bidder's console calls.

    POST /train   {"train_days": [0, 1], "eval_day": 2}   (both optional)
                  -> {"version": ..., "passed": ..., "log": [...]}
    POST /parity  -> the feature parity report (Chapter 8)
    GET  /healthz

One run at a time; a second request waits for the first. Deploying is the
bidder's job, not this service's: it checks parity and the validation
results itself before it swaps a model in.
"""
from __future__ import annotations

import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .train import Params, train


def serve(ledger_dir: str, models_dir: str, sim_out: str | None = None,
          host: str = "0.0.0.0", port: int = 8090) -> None:
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                return self._json(200, {"ok": True})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/parity":
                from . import parity
                try:
                    return self._json(200, parity.check(ledger_dir))
                except Exception as err:
                    traceback.print_exc()
                    return self._json(500, {"error": str(err)})
            if self.path != "/train":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                return self._json(400, {"error": "bad JSON"})
            lines = []

            def log(msg):
                print(msg, flush=True)
                lines.append(str(msg))

            p = Params()
            if req.get("eval_day") is not None:
                p.eval_day = int(req["eval_day"])
            if req.get("constant") is not None:
                p.constant = float(req["constant"])
            if req.get("train_days"):
                p.train_days = [int(d) for d in req["train_days"]]
            with lock:
                try:
                    version, rep = train(ledger_dir, models_dir, p,
                                         sim_out=sim_out, log=log)
                except Exception as err:
                    traceback.print_exc()
                    return self._json(500, {"error": str(err), "log": lines})
            self._json(200, {"version": version, "passed": rep["passed"],
                             "log": lines})

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"adtrain: training service on {host}:{port}, ledger={ledger_dir}, "
          f"models={models_dir}", flush=True)
    srv.serve_forever()
