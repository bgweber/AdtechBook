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
from pathlib import Path
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
            if self.path == "/labels":
                from . import labels
                try:
                    return self._json(200, labels.report(ledger_dir, sim_out))
                except Exception as err:
                    traceback.print_exc()
                    return self._json(500, {"error": str(err)})
            if self.path.startswith("/quarantine") or self.path == "/device_history":
                # quarantine (Chapter 24): the list, changes to it, and the
                # device history recounted without what it marks
                from . import quarantine as qr
                ddir = str(Path(models_dir).parent / "datasets")
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    if self.path == "/quarantine/add":
                        e = qr.add(ddir, "postbacks", int(req["from_ms"]), int(req["to_ms"]),
                                   req.get("where") or {}, str(req.get("reason") or ""))
                        return self._json(200, {"added": e, "entries": qr.load(ddir)})
                    if self.path == "/quarantine/remove":
                        qr.remove(ddir, str(req.get("id")))
                    if self.path == "/quarantine/inspect":
                        return self._json(200, qr.inspect(ledger_dir, int(req["day"]),
                                                          str(req.get("field") or "match_type")))
                    if self.path == "/device_history":
                        return self._json(200, qr.device_history(ledger_dir, ddir))
                    return self._json(200, {"entries": qr.load(ddir)})
                except Exception as err:
                    traceback.print_exc()
                    return self._json(500, {"error": str(err)})
            if self.path == "/parity":
                from . import parity
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                    req = json.loads(self.rfile.read(n) or b"{}")
                    return self._json(200, parity.check(ledger_dir, days=req.get("days")))
                except Exception as err:
                    traceback.print_exc()
                    return self._json(500, {"error": str(err)})
            if self.path in ("/replay", "/ope", "/periods", "/abtest", "/shadow", "/shadetest", "/lift"):
                # offline evaluation (Chapter 16): read-only, no lock needed
                from . import evaluate as ev
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    if self.path == "/replay":
                        out = ev.replay(ledger_dir, models_dir, int(req["day"]),
                                        str(req["b"]), str(req.get("a") or "logged"), sim_out)
                    elif self.path == "/ope":
                        out = ev.ope(ledger_dir, str(req.get("policy") or "uniform"),
                                     str(req.get("reward") or "click"))
                    elif self.path == "/abtest":
                        out = ev.abtest(ledger_dir, str(req["experiment"]), sim_out,
                                        goal=str(req.get("goal") or "CPI").upper())
                    elif self.path == "/lift":
                        from . import lift
                        hs = req.get("holdout_share")
                        out = lift.measure(ledger_dir, sim_out, int(req.get("days") or 28),
                                           float(hs) if hs else None, req.get("app") or None)
                    elif self.path == "/shadetest":
                        out = ev.shadetest(ledger_dir, models_dir, str(req["experiment"]))
                    elif self.path == "/shadow":
                        out = ev.shadow(ledger_dir, models_dir, str(req["experiment"]))
                    else:
                        out = ev.periods(ledger_dir, sim_out)
                except Exception as err:
                    traceback.print_exc()
                    return self._json(400, {"error": str(err)})
                return self._json(200, out)
            if self.path == "/pace":
                from . import pacing
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(n) or b"{}")
                    budgets = req.get("budgets") or {}
                    if not budgets:
                        # a bidder that just restarted has no campaign book
                        # until the simulator pushes it: use the simulator's
                        from . import simout
                        camp = simout.table(sim_out, "logs", "campaigns")
                        if camp is not None:
                            budgets = {str(c): float(b) for c, b in zip(camp.campaign_id, camp.daily_budget)}
                    f = pacing.forecast(ledger_dir, budgets, req.get("day"))
                    pacing.publish(models_dir, f)
                except Exception as err:
                    traceback.print_exc()
                    return self._json(400, {"error": str(err)})
                return self._json(200, f)
            if self.path == "/kappa":
                from . import kappa
                try:
                    c = kappa.compute(ledger_dir)
                    kappa.publish(models_dir, c)
                except Exception as err:
                    traceback.print_exc()
                    return self._json(400, {"error": str(err)})
                return self._json(200, c)
            if self.path not in ("/train", "/dataset", "/gbm", "/pipeline", "/win"):
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

            datasets_dir = str(Path(models_dir).parent / "datasets")
            if self.path == "/dataset":
                # a conversion dataset (Chapter 14); leak=true for the exercise
                from . import datasets
                with lock:
                    try:
                        df, m = datasets.build(ledger_dir, datasets.Settings(
                            leak=bool(req.get("leak")),
                            sample_rate=float(req.get("sample_rate") or 1.0)), log=log)
                        d = datasets.write(df, m, datasets_dir)
                    except Exception as err:
                        traceback.print_exc()
                        return self._json(500, {"error": str(err), "log": lines})
                m = datasets.read(d)[1]
                return self._json(200, {"version": m["version"], "passed": m["passed"],
                                        "checks": m["checks"], "log": lines})
            if self.path == "/win":
                # a win rate model on the newest days (Chapter 20)
                from . import win
                with lock:
                    try:
                        _, rep = win.train(ledger_dir, models_dir, use_mtw=req.get("use_mtw", True),
                                           sim_out=sim_out, log=log,
                                           curve=bool(req.get("curve") or req.get("curse")),
                                           curse=bool(req.get("curse")),
                                           sample_rate=float(req.get("sample_rate") or 1.0))
                    except Exception as err:
                        traceback.print_exc()
                        return self._json(500, {"error": str(err), "log": lines})
                return self._json(200, rep)
            if self.path == "/pipeline":
                # the whole conversion pipeline (Chapter 18)
                from . import pipeline
                inject = req.get("inject") or None
                if inject not in (None, "no-features"):
                    return self._json(400, {"error": "unknown injection"})
                with lock:
                    try:
                        rec = pipeline.run(ledger_dir, models_dir, sim_out,
                                           leak=bool(req.get("leak")), inject=inject, log=log)
                    except Exception as err:
                        traceback.print_exc()
                        return self._json(500, {"error": str(err), "log": lines})
                return self._json(200, rec)
            if self.path == "/gbm":
                # LightGBM conversion models on a dataset (Chapter 15)
                from . import gbm
                v = str(req.get("dataset") or "")
                if not v or "/" in v or v.startswith("."):
                    return self._json(400, {"error": "name a dataset version"})
                with lock:
                    try:
                        d, rep = gbm.train(Path(datasets_dir) / "conversions" / v,
                                           models_dir, sim_out, log=log)
                    except Exception as err:
                        traceback.print_exc()
                        return self._json(500, {"error": str(err), "log": lines})
                return self._json(200, {"version": rep["version"], "log": lines})

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
