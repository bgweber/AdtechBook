// Package console serves the admin console on the admin port: a live view
// of the bidder, its campaigns, and from later chapters its models,
// alerts, and controls.
//
// The bid port is only for exchanges. Anything a person, or the campaign
// management system, looks at or changes goes through this separate port,
// as in a production DSP, so admin traffic can never slow down bidding.
package console

import (
	"embed"
	"encoding/json"
	"fmt"
	"github.com/bgweber/adtechbook/code/dsp/internal/experiment"
	"github.com/bgweber/adtechbook/code/dsp/internal/policy"
	"io"
	"io/fs"
	"net/http"
	"net/http/pprof"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/features"
	"github.com/bgweber/adtechbook/code/dsp/internal/gbm"
	"github.com/bgweber/adtechbook/code/dsp/internal/latency"
	"github.com/bgweber/adtechbook/code/dsp/internal/ledger"
	"github.com/bgweber/adtechbook/code/dsp/internal/model"
	"github.com/bgweber/adtechbook/code/dsp/internal/ops"
	"github.com/bgweber/adtechbook/code/dsp/internal/profile"
	"github.com/bgweber/adtechbook/code/dsp/internal/stats"
)

//go:embed static
var static embed.FS

// Info is static information about the running bidder.
type Info struct {
	Version  string  `json:"version"`
	Mode     string  `json:"mode"`
	Stage    string  `json:"stage"`
	BidShare float64 `json:"bid_share"`
}

// Deps are the parts of the bidder the console reads or changes.
type Deps struct {
	Info      Info
	Stats     *stats.Stats
	Book      *campaigns.Book
	Profiles  *profile.Store
	Bids      *ledger.Writer
	Notices   *ledger.Writer
	Clicks    *ledger.Writer
	Postbacks *ledger.Writer
	Models    *model.Registry
	// Part III: LightGBM conversion bundles, and the conversion datasets
	// directory the training service writes.
	Conversion *gbm.Registry
	Datasets   string
	Ops        *ops.Ops
	Trainer    *Trainer
	Policy     *policy.Crawl       // the selection policy, switchable (Chapter 16)
	Exp        *experiment.Manager // shadow, canary, A/B (Chapter 19)
	Win        *gbm.Registry       // win rate models (Chapter 20)
	Uplift     *gbm.Registry       // uplift models (Chapter 32)
	Levers     string              // where the campaigns' levers are saved (Chapter 33)
	// Serving latency (Chapter 17): live requests' stage timings, the load
	// test's, and the file the load generator appends its runs to.
	Latency, LoadLatency *latency.Recorder
	LoadTests            string
}

// Handler returns the admin console's routes.
func Handler(d Deps) http.Handler {
	mux := http.NewServeMux()
	sub, _ := fs.Sub(static, "static")
	mux.Handle("/", http.FileServer(http.FS(sub)))
	mux.HandleFunc("/api/stats", func(w http.ResponseWriter, r *http.Request) {
		snap := d.Stats.Snapshot()
		day := int(max(snap.SimDay, 0))
		cur := ""
		if m := d.Models.Current(); m != nil {
			cur = m.Version
		}
		conv := ""
		if b := d.Conversion.Current(); b != nil {
			conv = b.Spec.Version
		}
		writeJSON(w, map[string]any{
			"info":       d.Info,
			"model":      cur,
			"conversion": conv,
			"stats":      snap,
			"campaigns":  d.Book.Views(day),
			"devices":    d.Profiles.Devices(),
			"ledger": map[string]int64{
				"bids_written": d.Bids.Written(), "bids_dropped": d.Bids.Dropped(),
				"notices_written": d.Notices.Written(), "notices_dropped": d.Notices.Dropped(),
				"clicks_written": d.Clicks.Written(), "clicks_dropped": d.Clicks.Dropped(),
				"postbacks_written": d.Postbacks.Written(), "postbacks_dropped": d.Postbacks.Dropped(),
			},
		})
	})
	// The campaign management system pushes the campaign book here. In
	// the simulator, adsim does it when a live run starts.
	mux.HandleFunc("/api/campaigns", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPut, http.MethodPost:
			body, err := io.ReadAll(io.LimitReader(r.Body, 16<<20))
			before := map[int]campaigns.Campaign{}
			for _, v := range d.Book.Views(0) {
				before[v.ID] = v.Campaign
			}
			if err == nil {
				err = d.Book.LoadJSON(body)
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			// A configuration change is a deploy: record what changed.
			sim := d.Stats.Snapshot().SimTimeMS
			changed := 0
			for _, v := range d.Book.Views(0) {
				old, ok := before[v.ID]
				switch {
				case len(before) == 0:
				case !ok:
					d.Ops.Note(fmt.Sprintf("campaign %d added", v.ID), "campaign system", sim)
				case old.Target != v.Target:
					d.Ops.Note(fmt.Sprintf("campaign %d target changed from %g to %g", v.ID, old.Target, v.Target),
						"campaign system", sim)
				case old.DailyBudget != v.DailyBudget:
					d.Ops.Note(fmt.Sprintf("campaign %d daily budget changed from %g to %g", v.ID,
						old.DailyBudget, v.DailyBudget), "campaign system", sim)
				default:
					continue
				}
				changed++
			}
			if len(before) == 0 {
				d.Ops.Note(fmt.Sprintf("campaign book loaded: %d campaigns (version %d)",
					len(d.Book.Views(0)), d.Book.Version()), "campaign system", sim)
			}
			writeJSON(w, map[string]int{"version": d.Book.Version(), "campaigns": len(d.Book.Views(0))})
		default:
			writeJSON(w, d.Book.Views(0))
		}
	})
	// Models (Chapters 9 and 11): the published click models, each one's
	// validation report, and deploys, including forced deploys and
	// rollback to the priors.
	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, d.Models.List())
	})
	mux.HandleFunc("/api/models/report", func(w http.ResponseWriter, r *http.Request) {
		raw, err := d.Models.Report(r.URL.Query().Get("version"))
		if err != nil {
			http.Error(w, "no such model", http.StatusNotFound)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		w.Write(raw)
	})
	mux.HandleFunc("/api/models/deploy", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Version string `json:"version"`
			Force   bool   `json:"force"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, d.Models.Deploy(req.Version, req.Force))
	})
	mux.HandleFunc("/api/train", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var req TrainRequest
			json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req)
			if err := d.Trainer.Start(req); err != nil {
				http.Error(w, err.Error(), http.StatusConflict)
				return
			}
		}
		st := d.Trainer.Status()
		writeJSON(w, map[string]any{"job": st, "available": d.Trainer.URL != ""})
	})
	// The simulator asks where to start (adsim --start-day auto), so a new
	// run continues the market rather than replaying days in the ledger.
	mux.HandleFunc("/api/ledger", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]int64{"next_day": d.Bids.NextDay()})
	})
	// Operations (Chapter 12): kill switches, throttle rates, alerts,
	// reconciliation, and the audit log.
	mux.HandleFunc("/api/ops", func(w http.ResponseWriter, r *http.Request) {
		sim := d.Stats.Snapshot().SimTimeMS
		d.Ops.Step(sim, d.Book) // keep alerts fresh while no traffic arrives
		rates, plan := d.Ops.RatesView()
		writeJSON(w, map[string]any{
			"switches": d.Ops.Switches(), "alerts": d.Ops.Alerts(), "audit": d.Ops.Audit(),
			"throttle":       map[string]any{"on": d.Ops.Throttle, "rates": rates, "planned_share": plan},
			"reconciliation": d.Ops.Reconciliation(),
			"features":       featureNames(d.Models),
		})
	})
	mux.HandleFunc("/api/ops/switch", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var req struct {
			Kind string `json:"kind"`
			Key  string `json:"key"`
			On   bool   `json:"on"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		sw, err := d.Ops.Set(req.Kind, req.Key, req.On, "console", d.Stats.Snapshot().SimTimeMS)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, sw)
	})
	// Exchanges send their invoices here; in the simulator, adsim does it
	// when a run ends.
	mux.HandleFunc("/api/invoices", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut && r.Method != http.MethodPost {
			writeJSON(w, d.Ops.Reconciliation())
			return
		}
		var lines []ops.InvoiceLine
		if err := json.NewDecoder(io.LimitReader(r.Body, 16<<20)).Decode(&lines); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		n := d.Ops.LoadInvoices(lines)
		exch := map[string]bool{}
		for _, l := range lines {
			exch[l.Exchange] = true
		}
		names := make([]string, 0, len(exch))
		for e := range exch {
			names = append(names, e)
		}
		d.Ops.Note(fmt.Sprintf("invoices received: %d lines from %s", n, strings.Join(names, ", ")),
			"exchanges", d.Stats.Snapshot().SimTimeMS)
		writeJSON(w, map[string]int{"lines": n})
	})
	// Conversion datasets (Chapter 14): GET lists the versions from their
	// manifests; POST builds one through the training service.
	mux.HandleFunc("/api/datasets", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var body struct {
				Leak bool `json:"leak"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			raw, err := d.Trainer.Call("dataset", body)
			reply(w, raw, err)
			return
		}
		writeJSON(w, listDatasets(d.Datasets))
	})
	// LightGBM conversion models (Chapter 15): GET lists the bundles;
	// POST /train trains on a dataset; POST /deploy and /undeploy.
	mux.HandleFunc("/api/conversion", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"bundles": d.Conversion.List()})
	})
	mux.HandleFunc("/api/conversion/train", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Dataset string `json:"dataset"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		raw, err := d.Trainer.Call("gbm", body)
		reply(w, raw, err)
	})
	mux.HandleFunc("/api/conversion/deploy", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			Version string `json:"version"`
			Force   bool   `json:"force"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		res := d.Conversion.Deploy(body.Version, body.Force)
		if res.OK {
			d.Ops.Note("conversion models "+body.Version+" deployed", "console", 0)
		}
		writeJSON(w, res)
	})
	// The pipeline and the champion/challenger registry (Chapter 18).
	mux.HandleFunc("/api/conversion/pipeline", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var body struct {
				Leak   bool   `json:"leak"`
				Inject string `json:"inject,omitempty"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			raw, err := d.Trainer.Call("pipeline", body)
			reply(w, raw, err)
			return
		}
		writeJSON(w, listRuns(filepath.Join(d.Conversion.Dir, gbm.Family, "pipeline"), 20))
	})
	// Incident drills (Chapter 24): GET lists them with the newest daily
	// parity result; POST /api/drills/start {"kind": "shift", "factor":
	// 1.1} or {"kind": "parity"}; POST /api/drills/stop {"kind": ...}.
	mux.HandleFunc("/api/drills", func(w http.ResponseWriter, r *http.Request) {
		serving := ""
		if b := d.Conversion.Current(); b != nil {
			serving = b.Spec.Version
		}
		writeJSON(w, map[string]any{"drills": d.Ops.Drills(), "parity": d.Ops.LastParity(),
			"serving": serving, "shift": d.Ops.ShiftView()})
	})
	mux.HandleFunc("/api/drills/start", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Kind   string  `json:"kind"`
			Factor float64 `json:"factor"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		switch body.Kind {
		case "shift":
			b := d.Conversion.Current()
			if b == nil {
				http.Error(w, "no conversion bundle is serving: deploy one first", http.StatusBadRequest)
				return
			}
			if body.Factor <= 0 {
				body.Factor = 1.1
			}
			d.Ops.StartShiftDrill(b.Spec.Version, body.Factor)
		case "parity":
			features.SkewFineConnection.Store(true)
			d.Ops.StartParityDrill("connection_type")
		default:
			http.Error(w, "unknown drill: shift or parity", http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"drills": d.Ops.Drills()})
	})
	mux.HandleFunc("/api/drills/stop", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Kind string `json:"kind"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if body.Kind == "parity" {
			features.SkewFineConnection.Store(false)
		}
		d.Ops.StopDrill(body.Kind)
		writeJSON(w, map[string]any{"drills": d.Ops.Drills()})
	})
	// Quarantine (Chapter 24), kept by the training service: GET the list;
	// POST /api/quarantine/add {"from_ms": ..., "to_ms": ..., "where":
	// {"match_type": "probabilistic"}, "reason": ...}, /remove {"id": ...},
	// /inspect {"day": 41}.
	for _, name := range []string{"", "/add", "/remove", "/inspect"} {
		name := name
		mux.HandleFunc("/api/quarantine"+name, func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			raw, err := d.Trainer.Call("quarantine"+name, body)
			if err == nil && name == "/add" {
				d.Ops.Note("quarantine added: "+string(raw), "console", 0)
			}
			reply(w, raw, err)
		})
	}
	// Recount every device's installs and purchases from the postbacks,
	// without the quarantined ones (Chapter 24, step 4).
	mux.HandleFunc("/api/profiles/recount", func(w http.ResponseWriter, r *http.Request) {
		raw, err := d.Trainer.Call("device_history", map[string]any{})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		var h struct {
			Devices map[string][2]int64 `json:"devices"`
			Skipped map[string]int64    `json:"skipped"`
		}
		if err := json.Unmarshal(raw, &h); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		changed := d.Profiles.SetConversions(h.Devices)
		d.Ops.Note(fmt.Sprintf("device profiles recounted from postbacks: %d devices changed", changed),
			"console", 0)
		writeJSON(w, map[string]any{"changed": changed, "devices": d.Profiles.Devices(), "skipped": h.Skipped})
	})
	// The daily parity job, on demand: POST {"day": 41}, or {} for
	// yesterday.
	mux.HandleFunc("/api/parity/day", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Day *int64 `json:"day"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		day := d.Stats.Snapshot().SimDay - 1
		if body.Day != nil {
			day = *body.Day
		}
		res, err := checkParity(d, day)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		writeJSON(w, res)
	})
	// Serving latency (Chapter 17): per-stage percentiles for live and
	// load-test requests, and the load generator's runs. POST
	// /api/latency/reset {"which": "load"} clears one recorder; the load
	// generator does before each rate.
	mux.HandleFunc("/api/latency", func(w http.ResponseWriter, r *http.Request) {
		var runs []json.RawMessage
		if b, err := os.ReadFile(d.LoadTests); err == nil {
			json.Unmarshal(b, &runs)
		}
		writeJSON(w, map[string]any{"live": d.Latency.Summary(), "load": d.LoadLatency.Summary(),
			"runs": runs, "auction_cap": d.Policy.AuctionCap.Load(), "cap_audit": d.Policy.CapAuditResult()})
	})
	mux.HandleFunc("/api/latency/reset", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Which string `json:"which"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if body.Which == "live" {
			d.Latency.Reset()
		} else {
			d.LoadLatency.Reset()
		}
		writeJSON(w, map[string]bool{"ok": true})
	})
	// The auction's candidate cap (Chapter 17): POST {"cap": 4} to price
	// only the four most promising campaigns by a cheap estimate; 0 prices
	// them all.
	mux.HandleFunc("/api/policy/auction-cap", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Cap int32 `json:"cap"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		d.Policy.AuctionCap.Store(max(body.Cap, 0))
		d.Policy.ResetCapAudit()
		d.Ops.Note(fmt.Sprintf("auction candidate cap: %d (0 = none)", max(body.Cap, 0)), "console", 0)
		writeJSON(w, map[string]any{"auction_cap": d.Policy.AuctionCap.Load()})
	})
	// The campaign selection policy (Chapter 16): GET the current one and
	// the choices; POST {"name": "uniform"} to switch.
	mux.HandleFunc("/api/policy/selection", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var body struct {
				Name string `json:"name"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			if !d.Policy.SetSelection(body.Name) {
				http.Error(w, "unknown selection policy", http.StatusBadRequest)
				return
			}
			d.Ops.Note("campaign selection: "+body.Name, "console", 0)
		}
		writeJSON(w, map[string]any{"current": d.Policy.SelectionName(), "choices": policy.Selections})
	})
	// Offline evaluation (Chapter 16), run by the training service.
	for _, name := range []string{"replay", "ope", "periods", "abtest", "shadow", "shadetest", "lift"} {
		name := name
		mux.HandleFunc("/api/eval/"+name, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "POST", http.StatusMethodNotAllowed)
				return
			}
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			raw, err := d.Trainer.Call(name, body)
			reply(w, raw, err)
		})
	}
	// Pricing for goals (Chapter 21): calibration multipliers by campaign,
	// and Thompson sampling over creatives.
	mux.HandleFunc("/api/calibration", func(w http.ResponseWriter, r *http.Request) {
		dir := filepath.Join(d.Conversion.Dir, "calibration")
		if r.Method == http.MethodPost {
			var body struct {
				Off bool `json:"off"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			if body.Off {
				d.Policy.Kappa.Off()
				d.Ops.Note("calibration multipliers off", "console", 0)
			} else {
				if _, err := d.Trainer.Call("kappa", map[string]any{}); err != nil {
					http.Error(w, err.Error(), http.StatusBadGateway)
					return
				}
				c, err := d.Policy.Kappa.Load(dir)
				if err != nil || c == nil {
					http.Error(w, fmt.Sprint("loading the calibration: ", err), http.StatusInternalServerError)
					return
				}
				d.Ops.Note("calibration "+c.Version+" loaded", "console", 0)
			}
		}
		var detail json.RawMessage
		if c := d.Policy.Kappa.Current(); c != nil {
			detail, _ = os.ReadFile(filepath.Join(dir, c.Version+".json"))
		}
		writeJSON(w, map[string]any{"current": d.Policy.Kappa.Current(), "detail": detail})
	})
	mux.HandleFunc("/api/creatives", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var body struct {
				Thompson bool `json:"thompson"`
				Reset    bool `json:"reset"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			if body.Reset {
				d.Policy.Creatives.Reset()
			}
			d.Policy.Creatives.SetThompson(body.Thompson)
			d.Ops.Note(fmt.Sprintf("Thompson sampling over creatives: %v", body.Thompson), "console", 0)
		}
		writeJSON(w, map[string]any{"thompson": d.Policy.Creatives.Thompson(),
			"campaigns": d.Policy.Creatives.Report()})
	})
	// Pacing (Chapter 22): the controller's state; POST {"on": true} to
	// switch from throttling; /overbid sets a manual overbid; /forecast
	// asks the training service for tomorrow's starting multipliers.
	mux.HandleFunc("/api/pacing", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			var body struct {
				On bool `json:"on"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			d.Ops.SetPacing(body.On)
			d.Ops.Note(fmt.Sprintf("pacing controller: %v (throttling: %v)", body.On, !body.On), "console", 0)
		}
		on, st := d.Ops.PaceView()
		day := int(max(d.Stats.Snapshot().SimDay, 0))
		writeJSON(w, map[string]any{"on": on, "campaigns": st, "book": d.Book.Views(day),
			"kp": ops.PaceKp, "ki": ops.PaceKi, "min": ops.PaceMin, "max": ops.PaceMax})
	})
	mux.HandleFunc("/api/pacing/overbid", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Campaign int     `json:"campaign"`
			Overbid  float64 `json:"overbid"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		d.Ops.SetOverbid(body.Campaign, body.Overbid)
		d.Ops.Note(fmt.Sprintf("campaign %d: manual overbid x%.2f", body.Campaign, body.Overbid), "console", 0)
		writeJSON(w, map[string]bool{"ok": true})
	})
	mux.HandleFunc("/api/pacing/forecast", func(w http.ResponseWriter, r *http.Request) {
		day := int(max(d.Stats.Snapshot().SimDay, 0))
		budgets := map[string]float64{}
		for _, v := range d.Book.Views(day) {
			budgets[strconv.Itoa(v.ID)] = v.DailyBudget
		}
		raw, err := d.Trainer.Call("pace", map[string]any{"budgets": budgets})
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		var f struct {
			Day int64 `json:"day"`
			By  map[string]struct {
				Gamma0 float64 `json:"gamma0"`
			} `json:"by_campaign"`
		}
		if err := json.Unmarshal(raw, &f); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		g := map[int]float64{}
		for k, v := range f.By {
			if id, err := strconv.Atoi(k); err == nil {
				g[id] = v.Gamma0
			}
		}
		d.Ops.SetForecast(f.Day, g)
		d.Ops.Note(fmt.Sprintf("pacing forecast for day %d loaded", f.Day), "console", 0)
		reply(w, raw, nil)
	})
	// Win rate models (Chapter 20): GET lists them; POST /train trains one
	// on the newest days of the ledger; /deploy shades every bid with one.
	mux.HandleFunc("/api/win", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"models": d.Win.List()})
	})
	mux.HandleFunc("/api/win/train", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		raw, err := d.Trainer.Call("win", body)
		reply(w, raw, err)
	})
	mux.HandleFunc("/api/win/deploy", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Version string `json:"version"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		res := d.Win.Deploy(body.Version, false)
		if res.OK {
			d.Ops.Note("win model "+body.Version+" deployed: bids shaded by expected surplus", "console", 0)
		}
		writeJSON(w, res)
	})
	// Uplift models (Chapter 32): GET lists them; POST /train trains one on
	// the lift study's devices. They're tested with a tilt A/B test, not
	// deployed.
	mux.HandleFunc("/api/uplift", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"models": d.Uplift.List()})
	})
	mux.HandleFunc("/api/uplift/train", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		raw, err := d.Trainer.Call("uplift", body)
		reply(w, raw, err)
	})
	// The unified pricing equation (Chapter 33): each campaign's levers, a
	// logged bid explained term by term, and the consistency check.
	mux.HandleFunc("/api/pricing/levers", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeJSON(w, d.Policy.Levers())
			return
		}
		var req struct {
			CampaignID int `json:"campaign_id"`
			policy.Levers
			Uplift *string `json:"uplift"`
			Clear  bool    `json:"clear"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<16)).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		cur := d.Policy.Levers()
		next := &policy.LeverSet{Campaigns: map[int]policy.Levers{}, Uplift: cur.Uplift}
		for k, v := range cur.Campaigns {
			next.Campaigns[k] = v
		}
		if req.Clear {
			delete(next.Campaigns, req.CampaignID)
		} else {
			if err := req.Levers.Check(); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			next.Campaigns[req.CampaignID] = req.Levers
		}
		if req.Uplift != nil && *req.Uplift != cur.Uplift {
			if err := d.Uplift.Pin(*req.Uplift); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			next.Uplift = *req.Uplift
		}
		if err := d.Policy.SetLevers(next, d.Levers); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		d.Ops.Note(fmt.Sprintf("campaign %d levers: %+v", req.CampaignID, req.Levers), "console", 0)
		writeJSON(w, next)
	})
	for _, name := range []string{"explain", "frontier", "rlpace"} {
		name := name
		mux.HandleFunc("/api/pricing/"+name, func(w http.ResponseWriter, r *http.Request) {
			var body map[string]any
			json.NewDecoder(r.Body).Decode(&body)
			raw, err := d.Trainer.Call(name, body)
			reply(w, raw, err)
		})
	}
	mux.HandleFunc("/api/win/undeploy", func(w http.ResponseWriter, r *http.Request) {
		d.Win.Undeploy()
		d.Ops.Note("win model switched off: fixed shade", "console", 0)
		writeJSON(w, map[string]bool{"ok": true})
	})
	// Experiments (Chapter 19). Times are simulated, from the traffic.
	mux.HandleFunc("/api/experiment", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, map[string]any{"current": d.Exp.Current(), "past": d.Exp.List()})
	})
	mux.HandleFunc("/api/experiment/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST", http.StatusMethodNotAllowed)
			return
		}
		now := d.Stats.Snapshot().SimTimeMS
		var err error
		switch strings.TrimPrefix(r.URL.Path, "/api/experiment/") {
		case "start":
			var cfg experiment.Config
			if err = json.NewDecoder(r.Body).Decode(&cfg); err == nil {
				_, err = d.Exp.Start(cfg, now)
			}
		case "stop":
			err = d.Exp.Stop(now, "stopped from the console")
		case "inject":
			var body struct {
				Fault string `json:"fault"`
			}
			json.NewDecoder(r.Body).Decode(&body)
			err = d.Exp.Inject(body.Fault, now)
		default:
			http.NotFound(w, r)
			return
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"current": d.Exp.Current()})
	})
	mux.HandleFunc("/api/conversion/history", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, d.Conversion.History(50))
	})
	for name, fn := range map[string]func() gbm.DeployResult{
		"promote": d.Conversion.Promote, "rollback": d.Conversion.Rollback} {
		name, fn := name, fn
		mux.HandleFunc("/api/conversion/"+name, func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost {
				http.Error(w, "POST", http.StatusMethodNotAllowed)
				return
			}
			res := fn()
			if res.OK {
				d.Ops.Note("conversion models: "+res.Message, "console "+name, 0)
			}
			writeJSON(w, res)
		})
	}
	mux.HandleFunc("/api/conversion/undeploy", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST", http.StatusMethodNotAllowed)
			return
		}
		d.Conversion.Undeploy()
		d.Ops.Note("conversion models switched off", "console", 0)
		writeJSON(w, map[string]bool{"ok": true})
	})

	// Reports from the training service. POST runs one, GET returns the
	// latest: feature parity (Chapter 8) and install labels (Chapter 13).
	for _, name := range []string{"parity", "labels"} {
		name := name
		mux.HandleFunc("/api/"+name, func(w http.ResponseWriter, r *http.Request) {
			var raw json.RawMessage
			var err error
			if r.Method == http.MethodPost {
				raw, err = d.Trainer.Report(name)
			} else {
				raw = d.Trainer.LastReport(name)
			}
			if err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
			if raw == nil {
				raw = json.RawMessage("null")
			}
			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("Cache-Control", "no-store")
			w.Write(raw)
		})
	}
	// Go's profiler, on the admin port only (Chapter 17): for example
	//   go tool pprof http://localhost:8081/debug/pprof/profile?seconds=10
	// while the load test runs shows which functions use the CPU.
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("ok\n"))
	})
	return mux
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	json.NewEncoder(w).Encode(v)
}

// featureNames lists the features the deployed model uses, for the
// feature kill switches.
func featureNames(r *model.Registry) []string {
	m := r.Current()
	if m == nil {
		return []string{}
	}
	out := append([]string(nil), m.Spec.Strings...)
	for k := range m.Spec.NumericBuckets {
		out = append(out, k)
	}
	for k := range m.Spec.Sets {
		out = append(out, k)
	}
	return out
}

// listDatasets reads each conversion dataset version's manifest, newest
// first, for the console.
func listDatasets(dir string) []json.RawMessage {
	root := filepath.Join(dir, "conversions")
	ents, _ := os.ReadDir(root)
	out := []json.RawMessage{}
	for i := len(ents) - 1; i >= 0; i-- {
		raw, err := os.ReadFile(filepath.Join(root, ents[i].Name(), "manifest.json"))
		if err == nil && json.Valid(raw) {
			out = append(out, raw)
		}
	}
	return out
}

// listRuns returns the newest pipeline run records.
func listRuns(dir string, limit int) []json.RawMessage {
	ents, _ := os.ReadDir(dir)
	out := []json.RawMessage{}
	for i := len(ents) - 1; i >= 0 && len(out) < limit; i-- {
		raw, err := os.ReadFile(filepath.Join(dir, ents[i].Name()))
		if err == nil && json.Valid(raw) {
			out = append(out, raw)
		}
	}
	return out
}

// reply passes a training service result through, or its error.
func reply(w http.ResponseWriter, raw json.RawMessage, err error) {
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(raw)
}
