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
	"io"
	"io/fs"
	"net/http"
	"strings"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
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
	Ops       *ops.Ops
	Trainer   *Trainer
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
		writeJSON(w, map[string]any{
			"info":      d.Info,
			"model":     cur,
			"stats":     snap,
			"campaigns": d.Book.Views(day),
			"devices":   d.Profiles.Devices(),
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
	// Feature parity (Chapter 8): POST runs the check, GET returns the
	// latest report.
	mux.HandleFunc("/api/parity", func(w http.ResponseWriter, r *http.Request) {
		var raw json.RawMessage
		var err error
		if r.Method == http.MethodPost {
			raw, err = d.Trainer.Parity()
		} else {
			raw = d.Trainer.LastParity()
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
