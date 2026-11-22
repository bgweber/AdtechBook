package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Registry is the bidder's view of the published click models: the
// artifacts under <dir>/click/<version>/, which one is deployed, and the
// swap from one to the next.
//
// The deployed version is written to <dir>/click/DEPLOYED, so a restarted
// bidder comes back with the same model, and every deploy is appended to
// <dir>/click/deploys.jsonl.
type Registry struct {
	Dir string // the models directory, such as out/models

	cur atomic.Pointer[Model]
	mu  sync.Mutex // serializes deploys
}

// Family is the model family this registry serves.
const Family = "click"

func (r *Registry) root() string { return filepath.Join(r.Dir, Family) }

// Current is the deployed model, or nil when the bidder uses its priors.
// It's one atomic load, so the bid path never waits on a deploy.
func (r *Registry) Current() *Model {
	if r == nil {
		return nil
	}
	return r.cur.Load()
}

// Summary is one model in the console's list.
type Summary struct {
	Version   string         `json:"version"`
	CreatedAt string         `json:"created_at"`
	Deployed  bool           `json:"deployed"`
	Passed    bool           `json:"passed"`
	Metrics   map[string]any `json:"metrics"`
	Logged    map[string]any `json:"logged"`
	Stability map[string]any `json:"stability"`
	Data      map[string]any `json:"data"`
}

// List returns the published models, newest first.
func (r *Registry) List() []Summary {
	ents, _ := os.ReadDir(r.root())
	cur := r.Current()
	out := []Summary{}
	for _, e := range ents {
		if !e.IsDir() {
			continue
		}
		rep, err := r.Report(e.Name())
		if err != nil {
			continue // still being written, or not a model
		}
		var s Summary
		json.Unmarshal(rep, &s)
		s.Version = e.Name()
		s.Deployed = cur != nil && cur.Version == s.Version
		out = append(out, s)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Version > out[j].Version })
	return out
}

// Report returns a model's report.json, as the training code wrote it.
func (r *Registry) Report(version string) ([]byte, error) {
	if !validVersion(version) {
		return nil, fmt.Errorf("bad version %q", version)
	}
	return os.ReadFile(filepath.Join(r.root(), version, "report.json"))
}

// DeployResult says what a deploy did, for the console.
type DeployResult struct {
	OK           bool    `json:"ok"`
	Version      string  `json:"version"`
	Message      string  `json:"message"`
	GoldenN      int     `json:"golden_examples,omitempty"`
	GoldenMaxErr float64 `json:"golden_max_diff,omitempty"`
	Forced       bool    `json:"forced,omitempty"`
	WarmupUS     int64   `json:"warmup_us,omitempty"` // per prediction, after warmup
}

// Deploy loads a model, checks that the bidder reproduces the training
// code's predictions on its golden examples, checks the training-time
// validation results, and swaps it in. A model that failed validation,
// such as the cosine stability check, is only deployed with force: the
// manual intervention of Chapter 11. Parity can't be forced: a model the
// bidder can't reproduce is never served. An empty version undeploys,
// and the bidder goes back to its priors.
func (r *Registry) Deploy(version string, force bool) DeployResult {
	r.mu.Lock()
	defer r.mu.Unlock()
	res := DeployResult{Version: version, Forced: force}
	if version == "" {
		r.cur.Store(nil)
		r.record("", force)
		res.OK, res.Message = true, "undeployed: bidding with the placement priors"
		return res
	}
	if !validVersion(version) {
		res.Message = "bad version"
		return res
	}
	dir := filepath.Join(r.root(), version)
	m, err := Load(dir)
	if err != nil {
		res.Message = "load failed: " + err.Error()
		return res
	}
	m.Version = version
	n, worst, err := m.CheckGolden(dir)
	res.GoldenN, res.GoldenMaxErr = n, worst
	if err != nil {
		res.Message = "parity check failed: " + err.Error()
		return res
	}
	if worst > ParityTolerance {
		res.Message = fmt.Sprintf("parity check failed: bidder and training predictions "+
			"differ by up to %.3g on %d golden examples", worst, n)
		return res
	}
	var rep struct {
		Passed bool `json:"passed"`
		Checks []struct {
			Name string `json:"name"`
			Pass bool   `json:"pass"`
		} `json:"checks"`
	}
	raw, err := r.Report(version)
	if err == nil {
		err = json.Unmarshal(raw, &rep)
	}
	if err != nil {
		res.Message = "no readable report.json: " + err.Error()
		return res
	}
	if !rep.Passed && !force {
		var failed []string
		for _, c := range rep.Checks {
			if !c.Pass {
				failed = append(failed, c.Name)
			}
		}
		res.Message = "not deployed: failed " + strings.Join(failed, ", ") +
			". Review the model, then deploy with force to override."
		return res
	}
	// Warm up before the swap (Chapter 11): score the golden examples a
	// few times, so the first real requests don't pay for cold caches,
	// and check that one prediction fits easily in the latency budget.
	res.WarmupUS = m.warmup(dir)
	if res.WarmupUS > MaxPredictUS {
		res.Message = fmt.Sprintf("not deployed: a prediction takes %d µs after warmup, over the %d µs budget",
			res.WarmupUS, MaxPredictUS)
		return res
	}
	r.cur.Store(m)
	r.record(version, force)
	res.OK = true
	res.Message = fmt.Sprintf("deployed %s: %d nonzero weights, parity OK on %d golden examples, "+
		"%d µs per prediction after warmup", version, m.NonzeroWeights, n, res.WarmupUS)
	return res
}

// LoadDeployed restores the deployed model after a restart.
func (r *Registry) LoadDeployed() (string, error) {
	raw, err := os.ReadFile(filepath.Join(r.root(), "DEPLOYED"))
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	v := strings.TrimSpace(string(raw))
	if v == "" {
		return "", nil
	}
	res := r.Deploy(v, true) // it passed, or was forced, when first deployed
	if !res.OK {
		return v, fmt.Errorf("%s", res.Message)
	}
	return v, nil
}

func (r *Registry) record(version string, force bool) {
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
	json.NewEncoder(f).Encode(map[string]any{
		"at": time.Now().UTC().Format(time.RFC3339), "version": version, "forced": force})
}

// validVersion keeps a version from escaping the models directory.
func validVersion(v string) bool {
	if v == "" || len(v) > 64 {
		return false
	}
	for _, c := range v {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '-' || c == '_' || c == '.') {
			return false
		}
	}
	return v != "." && v != ".."
}

// MaxPredictUS is the most one prediction may take after warmup.
const MaxPredictUS = 2000

func (m *Model) warmup(dir string) int64 {
	f, err := os.Open(filepath.Join(dir, "golden.jsonl"))
	if err != nil {
		return 0
	}
	defer f.Close()
	var gs []Golden
	dec := json.NewDecoder(f)
	for {
		var g Golden
		if dec.Decode(&g) != nil {
			break
		}
		gs = append(gs, g)
	}
	if len(gs) == 0 {
		return 0
	}
	for i := 0; i < 3; i++ {
		for j := range gs {
			m.Predict(&gs[j].Features, gs[j].Ad)
		}
	}
	start := time.Now()
	for j := range gs {
		m.Predict(&gs[j].Features, gs[j].Ad)
	}
	return max(1, time.Since(start).Microseconds()/int64(len(gs)))
}
