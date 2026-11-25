package gbm

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// Registry is the bidder's view of the published conversion bundles under
// <dir>/conversion/<version>/: which one is deployed, and the swap from one
// to the next. It works like the click model registry of Chapter 11: the
// deployed version is kept in <dir>/conversion/DEPLOYED, every deploy is
// appended to deploys.jsonl, and a bundle whose golden examples the bidder
// can't reproduce is never deployed, forced or not.
type Registry struct {
	Dir    string // the models directory, such as out/models
	Family string // its subdirectory: conversion (the default) or win

	cur    atomic.Pointer[Bundle]
	pinned atomic.Pointer[Bundle] // an uplift bundle the campaigns' tilts use (Chapter 33)
	mu     sync.Mutex
}

// Family is the default directory a registry serves.
const Family = "conversion"

// Tolerance for the golden examples, as for the click model.
const Tolerance = 1e-9

// MaxPredictUS is the warmup limit: a bundle slower than this, for all of
// its models together, isn't deployed.
const MaxPredictUS = 2000

func (r *Registry) root() string {
	if r.Family != "" {
		return filepath.Join(r.Dir, r.Family)
	}
	return filepath.Join(r.Dir, Family)
}

// Current is the deployed bundle, or nil. One atomic load.
func (r *Registry) Current() *Bundle {
	if r == nil {
		return nil
	}
	return r.cur.Load()
}

// Summary is one bundle in the console's list: its report as the training
// code wrote it, and whether it's deployed.
type Summary struct {
	Version    string          `json:"version"`
	Deployed   bool            `json:"deployed"`   // the champion
	Challenger bool            `json:"challenger"` // passed the pipeline, not yet promoted
	Report     json.RawMessage `json:"report"`
	Validation json.RawMessage `json:"validation,omitempty"` // the pipeline's model checks
}

// List returns the published bundles, newest first.
func (r *Registry) List() []Summary {
	ents, _ := os.ReadDir(r.root())
	cur := r.Current()
	ch := r.Challenger()
	out := []Summary{}
	for _, e := range ents {
		if !e.IsDir() || !validVersion(e.Name()) {
			continue
		}
		rep, err := os.ReadFile(filepath.Join(r.root(), e.Name(), "report.json"))
		if err != nil {
			continue // still being written
		}
		val, _ := os.ReadFile(filepath.Join(r.root(), e.Name(), "validation.json"))
		if !json.Valid(val) {
			val = nil
		}
		out = append(out, Summary{Version: e.Name(), Report: rep, Validation: val,
			Deployed:   cur != nil && cur.Spec.Version == e.Name(),
			Challenger: ch == e.Name() && (cur == nil || cur.Spec.Version != ch)})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version > out[j].Version })
	return out
}

// DeployResult says what happened.
type DeployResult struct {
	OK        bool    `json:"ok"`
	Message   string  `json:"message"`
	Golden    int     `json:"golden"`
	WorstDiff float64 `json:"worst_diff"`
	WarmupUS  float64 `json:"warmup_us"`
}

// Challenger is the version the pipeline last published as the
// challenger (the CHALLENGER file, Chapter 18), or "".
func (r *Registry) Challenger() string {
	raw, _ := os.ReadFile(filepath.Join(r.root(), "CHALLENGER"))
	return strings.TrimSpace(string(raw))
}

// Deploy loads a bundle, checks it, and swaps it in. A bundle trained on a
// dataset that failed validation, or that failed the pipeline's model
// checks, needs force; the golden check and the warmup limit can't be
// forced.
func (r *Registry) Deploy(version string, force bool) DeployResult {
	return r.deploy(version, force, "deploy")
}

// Promote makes the challenger the champion: a deploy, logged as one.
func (r *Registry) Promote() DeployResult {
	ch := r.Challenger()
	if ch == "" {
		return DeployResult{Message: "no challenger: run the pipeline first"}
	}
	if cur := r.Current(); cur != nil && cur.Spec.Version == ch {
		return DeployResult{Message: ch + " is already the champion"}
	}
	return r.deploy(ch, false, "promote")
}

// Rollback goes back to the version that served before the current one,
// from the deploy log. That version served already, so its validation
// isn't asked again (force); its golden examples still are.
func (r *Registry) Rollback() DeployResult {
	hist := r.History(0)
	cur := ""
	if b := r.Current(); b != nil {
		cur = b.Spec.Version
	}
	for _, h := range hist {
		if h.Version != "" && h.Version != cur {
			return r.deploy(h.Version, true, "rollback")
		}
	}
	return DeployResult{Message: "nothing to roll back to: no earlier version in deploys.jsonl"}
}

// Event is one line of deploys.jsonl.
type Event struct {
	At       string `json:"at"`
	Version  string `json:"version"`
	Previous string `json:"previous,omitempty"`
	Action   string `json:"action,omitempty"` // deploy, promote, rollback, undeploy, restart
	Forced   bool   `json:"forced"`
}

// History returns the deploy log, newest first; limit 0 is all of it.
func (r *Registry) History(limit int) []Event {
	raw, _ := os.ReadFile(filepath.Join(r.root(), "deploys.jsonl"))
	var out []Event
	lines := strings.Split(strings.TrimSpace(string(raw)), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		var e Event
		if json.Unmarshal([]byte(lines[i]), &e) == nil {
			out = append(out, e)
		}
		if limit > 0 && len(out) == limit {
			break
		}
	}
	return out
}

// Prepare loads a bundle and runs every deploy check on it, without
// serving it: the experiments of Chapter 19 use it for a challenger.
func (r *Registry) Prepare(version string, force bool) (*Bundle, DeployResult) {
	if !validVersion(version) {
		return nil, DeployResult{Message: fmt.Sprintf("bad version %q", version)}
	}
	dir := filepath.Join(r.root(), version)
	b, err := Load(dir)
	if err != nil {
		return nil, DeployResult{Message: "can't load " + version + ": " + err.Error()}
	}
	b.Spec.Version = version // the registry's name for it, which the logs carry
	n, worst, err := b.CheckGolden(dir)
	res := DeployResult{Golden: n, WorstDiff: worst}
	if err != nil {
		res.Message = "golden examples: " + err.Error()
		return nil, res
	}
	if n == 0 || worst > Tolerance || math.IsNaN(worst) {
		res.Message = fmt.Sprintf("refused: the bidder's predictions differ from the trainer's "+
			"by up to %.3g on %d golden examples (limit %g); this can't be forced", worst, n, Tolerance)
		return nil, res
	}
	var rep struct {
		DatasetPassed *bool `json:"dataset_passed"`
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "report.json")); err == nil {
		json.Unmarshal(raw, &rep)
	}
	if rep.DatasetPassed != nil && !*rep.DatasetPassed && !force {
		res.Message = "refused: trained on a dataset that failed validation; deploy with force to override"
		return nil, res
	}
	var val struct {
		Checks []struct {
			Name   string `json:"name"`
			Passed bool   `json:"passed"`
		} `json:"checks"`
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "validation.json")); err == nil {
		json.Unmarshal(raw, &val)
	}
	for _, c := range val.Checks {
		if !c.Passed && !force {
			res.Message = "refused: failed the pipeline's check \"" + c.Name + "\"; deploy with force to override"
			return nil, res
		}
	}
	res.WarmupUS = b.warmup(dir)
	if res.WarmupUS > MaxPredictUS {
		res.Message = fmt.Sprintf("refused: %.0f µs per prediction after warmup (limit %d)",
			res.WarmupUS, MaxPredictUS)
		return nil, res
	}
	res.OK = true
	res.Message = fmt.Sprintf("%s: %d golden examples match to %.2g, %.1f µs per prediction",
		version, n, worst, res.WarmupUS)
	return b, res
}

func (r *Registry) deploy(version string, force bool, action string) DeployResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	b, res := r.Prepare(version, force)
	if b == nil {
		return res
	}
	prev := ""
	if old := r.cur.Load(); old != nil {
		prev = old.Spec.Version
	}
	r.cur.Store(b)
	if action != "restart" {
		r.record(version, prev, action, force)
	}
	res.Message = "deployed " + res.Message
	return res
}

// Undeploy switches the conversion models off: the bidder goes back to
// the click model and segment install rates.
func (r *Registry) Undeploy() {
	r.mu.Lock()
	defer r.mu.Unlock()
	prev := ""
	if old := r.cur.Load(); old != nil {
		prev = old.Spec.Version
	}
	r.cur.Store(nil)
	r.record("", prev, "undeploy", false)
}

// LoadDeployed restores the deployed bundle after a restart.
func (r *Registry) LoadDeployed() (string, error) {
	raw, err := os.ReadFile(filepath.Join(r.root(), "DEPLOYED"))
	if err != nil || strings.TrimSpace(string(raw)) == "" {
		return "", nil
	}
	v := strings.TrimSpace(string(raw))
	if res := r.deploy(v, true, "restart"); !res.OK {
		return v, fmt.Errorf("%s", res.Message)
	}
	return v, nil
}

// warmup runs the golden examples a few times, then times every model on
// them, and returns microseconds per example.
func (b *Bundle) warmup(dir string) float64 {
	var rows [][]float64
	f, err := os.ReadFile(filepath.Join(dir, "golden.jsonl"))
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(strings.TrimSpace(string(f)), "\n") {
		var g struct {
			Features features.FeatureVector `json:"features"`
			Ad       map[string]string      `json:"ad"`
		}
		if json.Unmarshal([]byte(line), &g) == nil {
			rows = append(rows, b.Inputs(&g.Features, g.Ad))
		}
	}
	if len(rows) == 0 {
		return 0
	}
	run := func() {
		for _, x := range rows {
			for _, m := range b.Models {
				m.Predict(x)
			}
		}
	}
	for i := 0; i < 3; i++ {
		run()
	}
	start := time.Now()
	run()
	return float64(time.Since(start).Microseconds()) / float64(len(rows))
}

func (r *Registry) record(version, prev, action string, force bool) {
	os.MkdirAll(r.root(), 0o755)
	tmp := filepath.Join(r.root(), "DEPLOYED.tmp")
	if os.WriteFile(tmp, []byte(version+"\n"), 0o644) == nil {
		os.Rename(tmp, filepath.Join(r.root(), "DEPLOYED"))
	}
	f, err := os.OpenFile(filepath.Join(r.root(), "deploys.jsonl"),
		os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	json.NewEncoder(f).Encode(Event{At: time.Now().UTC().Format(time.RFC3339),
		Version: version, Previous: prev, Action: action, Forced: force})
}

func validVersion(v string) bool {
	if v == "" || len(v) > 64 || strings.HasPrefix(v, ".") {
		return false
	}
	for _, c := range v {
		if !(c >= 'a' && c <= 'z' || c >= '0' && c <= '9' || c == '-' || c == '_') {
			return false
		}
	}
	return true
}
