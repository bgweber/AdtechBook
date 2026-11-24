// Package experiment runs one model experiment at a time next to the
// champion (Chapter 19):
//
//   - shadow: the challenger scores a sample of requests and logs its
//     predictions next to the champion's; the champion still bids.
//   - canary: devices are assigned to the challenger by a salted hash, a
//     small share at first, then more in stages. Guardrails compare the
//     two arms as the canary runs; a failed check rolls it back, and
//     passing every stage promotes it.
//   - ab: a fixed share of devices gets the challenger until the test is
//     stopped, with each campaign's budget split between the arms. The
//     outcome analysis is offline (adtrain abtest), once installs arrive.
//   - shading: an A/B test in which the challenger is a win rate model,
//     and the treatment shades bids with it instead of the fixed factor
//     (Chapter 20).
//   - value: an A/B test in which the treatment prices Ad ROAS campaigns
//     with the challenger bundle's value model, and everything else with
//     the champion (Chapter 30).
//   - curse: an A/B test in which the challenger is a win bundle with a
//     winner's curse table, and the treatment marks values down by their
//     margin over its predicted market price (Chapter 31).
//
// Every bid record carries the experiment's ID and the request's arm, so
// the analysis comes from the ledger. Each experiment's record is kept in
// <models>/experiments/exp-NNNN.json.
package experiment

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/gbm"
	"github.com/bgweber/adtechbook/code/dsp/internal/policy"
)

// Kinds of experiment.
const (
	Shadow  = "shadow"
	Canary  = "canary"
	AB      = "ab"
	Shading = "shading" // A/B test of model-based shading against the fixed shade (Chapter 20)
	Value   = "value"   // A/B test of a value model for Ad ROAS against the champion's (Chapter 30)
	Curse   = "curse"   // A/B test of a win bundle's winner's curse correction (Chapter 31)
)

// Guardrail limits for a canary.
const (
	MaxLogRatio   = 0.405 // |ln(mean canary prediction / mean control)|: 1.5x either way
	MinSegmentN   = 100   // canary decisions before a segment's ratio counts
	MinStageN     = 300   // canary decisions before a stage can pass
	MaxFallbackPP = 0.05  // canary fallback share, above the control's
	MaxSpendRatio = 2.0   // canary spend share over its traffic share
	MinSpend      = 1.0   // dollars, before the spend check counts
	CheckEveryMS  = 15 * 60 * 1000
)

// Config is what the console asks for.
type Config struct {
	Kind        string    `json:"kind"`
	Challenger  string    `json:"challenger"`             // a conversion bundle version
	Share       float64   `json:"share,omitempty"`        // shadow sample, or A/B treatment share
	Stages      []float64 `json:"stages,omitempty"`       // canary shares; passing the last promotes
	StageHours  float64   `json:"stage_hours,omitempty"`  // simulated hours per canary stage
	BudgetSplit bool      `json:"budget_split,omitempty"` // A/B: split each campaign's budget
	Force       bool      `json:"force,omitempty"`        // a challenger that failed validation
}

// Check is one guardrail, as last evaluated.
type Check struct {
	Name   string  `json:"name"`
	Passed bool    `json:"passed"`
	Value  float64 `json:"value"`
	Limit  float64 `json:"limit"`
	Detail string  `json:"detail"`
}

// Event is one line of an experiment's history.
type Event struct {
	TsMS int64  `json:"ts_ms"` // simulated time
	What string `json:"what"`
}

type seg struct {
	N        int64   `json:"decisions"`
	Click    float64 `json:"sum_p_click"`
	NInst    int64   `json:"install_decisions"`
	Install  float64 `json:"sum_p_install_click"`
	Fallback int64   `json:"fallbacks"`
}

func (s *seg) add(d *policy.Decision) {
	s.N++
	s.Click += d.PClick
	if d.PClickFrom != "model" {
		s.Fallback++
	}
	if d.PInstallClick > 0 && d.InstallRateFrom != "" {
		s.NInst++
		s.Install += d.PInstallClick
	}
}

// Paired compares the canary's predictions with the champion's on the
// same requests: the canary's own, scored by both. Different devices in
// each arm would make the comparison noisy; the same requests don't.
type Paired struct {
	N        int64   `json:"decisions"`
	Canary   float64 `json:"sum_p_click_canary"`
	Champion float64 `json:"sum_p_click_champion"`
	NInst    int64   `json:"install_decisions"`
	InstC    float64 `json:"sum_p_install_canary"`
	InstH    float64 `json:"sum_p_install_champion"`
}

// ArmStats are one arm's numbers, for the current stage (canary) or the
// whole experiment (shadow, A/B).
type ArmStats struct {
	Requests    int64              `json:"requests"`
	Bids        int64              `json:"bids"`
	Spend       float64            `json:"spend"`
	All         seg                `json:"all"`
	ByPlacement map[string]*seg    `json:"by_placement"`
	Paired      map[string]*Paired `json:"paired,omitempty"` // treatment only: by placement, and "all"
}

func newArm() *ArmStats {
	return &ArmStats{ByPlacement: map[string]*seg{}, Paired: map[string]*Paired{}}
}

// ShadowStats compare the challenger's shadow predictions with the
// champion's on the same requests.
type ShadowStats struct {
	N          int64              `json:"scored"`
	Champion   map[string]float64 `json:"champion_mean_p_install_by_placement"`
	Challenger map[string]float64 `json:"challenger_mean_p_install_by_placement"`
	ClickRatio map[string]float64 `json:"click_ratio_by_placement"`
	P50US      float64            `json:"latency_p50_us"`
	P99US      float64            `json:"latency_p99_us"`

	sumC, sumS, nI map[string]float64
	clickC, clickS map[string]float64
	lat            []float64
}

// Experiment is one experiment's state and record.
type Experiment struct {
	ID           string       `json:"id"`
	Config       Config       `json:"config"`
	Status       string       `json:"status"` // running, rolled back, promoted, stopped, interrupted
	Reason       string       `json:"reason,omitempty"`
	StartedMS    int64        `json:"started_ms"`
	EndedMS      int64        `json:"ended_ms,omitempty"`
	StartedAt    string       `json:"started_at"`
	Stage        int          `json:"stage"`
	StageStartMS int64        `json:"stage_start_ms"`
	TreatShare   float64      `json:"treatment_share"`
	Fault        string       `json:"fault,omitempty"`
	Checks       []Check      `json:"checks,omitempty"`
	Events       []Event      `json:"events"`
	Arms         [2]*ArmStats `json:"arms"` // control, treatment
	Shadow       *ShadowStats `json:"shadow,omitempty"`

	lastCheckMS int64
}

// Live is what the bid path reads, once per request, without a lock.
type Live struct {
	ID     string
	Kind   string
	Share  float64 // treatment share now, or the shadow sample
	Salt   string
	Bundle *gbm.Bundle   // a conversion challenger
	Win    *gbm.WinModel // a win model challenger (shading tests)
	Fault  string
}

// Arm assigns a device: a salted hash, so a device stays in one arm.
func (l *Live) Arm(ifa string) int {
	if l == nil || l.Kind == Shadow {
		return campaigns.ArmNone
	}
	if bucket(l.Salt, ifa) < l.Share {
		return campaigns.ArmTreatment
	}
	return campaigns.ArmControl
}

// Sampled says whether shadow mode scores this request.
func (l *Live) Sampled(requestID string) bool {
	return l != nil && l.Kind == Shadow && bucket(l.Salt, requestID) < l.Share
}

func bucket(salt, key string) float64 {
	h := fnv.New64a()
	h.Write([]byte(salt + ":" + key))
	return float64(h.Sum64()%100000) / 100000
}

// Manager runs at most one experiment.
type Manager struct {
	Dir      string        // where experiment records go, such as out/models/experiments
	Registry *gbm.Registry // conversion bundles
	Win      *gbm.Registry // win models (Chapter 20)
	Book     *campaigns.Book
	Note     func(what string) // the ops audit log

	mu   sync.Mutex
	cur  *Experiment
	live atomic.Pointer[Live]
}

// Live returns the running experiment's bid-path view, or nil.
func (m *Manager) Live() *Live {
	if m == nil {
		return nil
	}
	return m.live.Load()
}

// Start begins an experiment. The challenger passes the same checks as a
// deploy first: an experiment is a deploy to part of the traffic.
func (m *Manager) Start(cfg Config, nowMS int64) (*Experiment, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur != nil && m.cur.Status == "running" {
		return nil, fmt.Errorf("%s is still running: stop it first", m.cur.ID)
	}
	switch cfg.Kind {
	case Shadow:
		if cfg.Share <= 0 || cfg.Share > 1 {
			cfg.Share = 0.1
		}
	case Canary:
		if len(cfg.Stages) == 0 {
			cfg.Stages = []float64{0.05, 0.25, 0.5}
		}
		if cfg.StageHours <= 0 {
			cfg.StageHours = 4
		}
	case AB, Shading, Value, Curse:
		if cfg.Share <= 0 || cfg.Share >= 1 {
			cfg.Share = 0.5
		}
	default:
		return nil, fmt.Errorf("kind must be shadow, canary, ab, shading, value, or curse")
	}
	reg := m.Registry
	if cfg.Kind == Shading || cfg.Kind == Curse {
		reg = m.Win
		if reg == nil {
			return nil, fmt.Errorf("no win model registry")
		}
	}
	// a value test can use the champion's own value model: the control arm
	// prices Ad ROAS with the bundle's per-app averages instead
	if cur := reg.Current(); cur != nil && cur.Spec.Version == cfg.Challenger && cfg.Kind != Value && cfg.Kind != Curse {
		return nil, fmt.Errorf("%s is already the champion", cfg.Challenger)
	}
	b, res := reg.Prepare(cfg.Challenger, cfg.Force)
	if b == nil {
		return nil, fmt.Errorf("%s", res.Message)
	}
	if cfg.Kind == Value {
		if _, ok := b.Models["value"]; !ok {
			return nil, fmt.Errorf("%s has no value model: train on a market with Ad ROAS campaigns", cfg.Challenger)
		}
	}
	var wm *gbm.WinModel
	if cfg.Kind == Shading || cfg.Kind == Curse {
		var err error
		if wm, err = gbm.NewWinModel(b); err != nil {
			return nil, err
		}
		if cfg.Kind == Curse && (b.Curse == nil || !wm.Curve()) {
			return nil, fmt.Errorf("%s has no win curve and curse table: train it with --curve --curse", cfg.Challenger)
		}
		b = nil
	}
	e := &Experiment{ID: m.nextID(), Config: cfg, Status: "running", StartedMS: nowMS,
		StartedAt: time.Now().UTC().Format(time.RFC3339), StageStartMS: nowMS,
		Arms: [2]*ArmStats{newArm(), newArm()}}
	switch cfg.Kind {
	case Shadow:
		e.TreatShare = 0
		e.Shadow = &ShadowStats{sumC: map[string]float64{}, sumS: map[string]float64{},
			nI: map[string]float64{}, clickC: map[string]float64{}, clickS: map[string]float64{}}
		e.event(nowMS, fmt.Sprintf("shadow: %s scores %.0f%% of requests; the champion bids", cfg.Challenger, 100*cfg.Share))
	case Canary:
		e.TreatShare = cfg.Stages[0]
		e.event(nowMS, fmt.Sprintf("canary: %s on %.0f%% of devices, %g simulated hours a stage",
			cfg.Challenger, 100*e.TreatShare, cfg.StageHours))
	case AB, Shading, Value, Curse:
		e.TreatShare = cfg.Share
		if cfg.BudgetSplit {
			m.Book.SetSplit(cfg.Share)
		}
		what := "A/B test: " + cfg.Challenger
		if cfg.Kind == Shading {
			what = "shading A/B test: bids shaded with " + cfg.Challenger + " instead of the fixed factor,"
		}
		if cfg.Kind == Value {
			what = "value A/B test: Ad ROAS bids priced with " + cfg.Challenger + "'s value model,"
		}
		if cfg.Kind == Curse {
			what = "winner's curse A/B test: values marked down by margin with " + cfg.Challenger + ","
		}
		e.event(nowMS, fmt.Sprintf("%s on %.0f%% of devices%s", what, 100*cfg.Share,
			map[bool]string{true: ", with each campaign's budget split", false: ", budgets shared"}[cfg.BudgetSplit]))
	}
	m.cur = e
	m.publish(b, wm)
	m.save()
	m.note(e.ID + " started: " + e.Events[0].What)
	return e, nil
}

func (m *Manager) publish(b *gbm.Bundle, wm *gbm.WinModel) {
	e := m.cur
	if e == nil || e.Status != "running" {
		m.live.Store(nil)
		return
	}
	if old := m.live.Load(); old != nil {
		if b == nil {
			b = old.Bundle
		}
		if wm == nil {
			wm = old.Win
		}
	}
	share := e.TreatShare
	if e.Config.Kind == Shadow {
		share = e.Config.Share
	}
	m.live.Store(&Live{ID: e.ID, Kind: e.Config.Kind, Share: share, Salt: e.ID,
		Bundle: b, Win: wm, Fault: e.Fault})
}

// Stop ends the running experiment.
func (m *Manager) Stop(nowMS int64, why string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur == nil || m.cur.Status != "running" {
		return fmt.Errorf("no experiment is running")
	}
	m.end(nowMS, "stopped", why)
	return nil
}

func (m *Manager) end(nowMS int64, status, why string) {
	e := m.cur
	e.Status, e.Reason, e.EndedMS = status, why, nowMS
	e.event(nowMS, status+": "+why)
	if (e.Config.Kind == AB || e.Config.Kind == Shading || e.Config.Kind == Value || e.Config.Kind == Curse) && e.Config.BudgetSplit {
		m.Book.SetSplit(0)
	}
	m.live.Store(nil)
	m.save()
	m.note(e.ID + " " + status + ": " + why)
}

// Inject breaks the challenger on purpose (Chapter 19): "scale" triples
// its click and install predictions from now on.
func (m *Manager) Inject(fault string, nowMS int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur == nil || m.cur.Status != "running" {
		return fmt.Errorf("no experiment is running")
	}
	if fault != "scale" {
		return fmt.Errorf("unknown fault %q", fault)
	}
	m.cur.Fault = fault
	m.cur.event(nowMS, "fault injected: the challenger's predictions are tripled")
	m.publish(nil, nil)
	m.save()
	return nil
}

// Observe records one decision, then runs the canary's checks every
// simulated quarter hour.
// champ is the champion's prediction for a treatment request, for the
// paired comparison; nil otherwise.
func (m *Manager) Observe(l *Live, arm int, placement string, d *policy.Decision,
	champ *policy.Prediction, tsMS int64) {
	if l == nil || l.Kind == Shadow || arm == campaigns.ArmNone {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.cur
	if e == nil || e.ID != l.ID || e.Status != "running" {
		return
	}
	e.clock(tsMS)
	a := e.Arms[arm]
	a.Requests++
	if d.CampaignID >= 0 && d.PClick > 0 {
		a.All.add(d)
		s := a.ByPlacement[placement]
		if s == nil {
			s = &seg{}
			a.ByPlacement[placement] = s
		}
		s.add(d)
	}
	if d.Action == "bid" {
		a.Bids++
	}
	if champ != nil && d.CampaignID >= 0 && d.PClick > 0 && champ.PClick > 0 {
		for _, k := range []string{"all", placement} {
			p := a.Paired[k]
			if p == nil {
				p = &Paired{}
				a.Paired[k] = p
			}
			p.N++
			p.Canary += d.PClick
			p.Champion += champ.PClick
			if d.PInstallClick > 0 && champ.PInstallClick > 0 && d.InstallRateFrom == "model" {
				p.NInst++
				p.InstC += d.PInstallClick
				p.InstH += champ.PInstallClick
			}
		}
	}
	if e.Config.Kind == Canary && tsMS-e.lastCheckMS >= CheckEveryMS {
		e.lastCheckMS = tsMS
		m.checkCanary(tsMS)
	}
}

// ObserveSpend records a billed impression for an arm.
func (m *Manager) ObserveSpend(id string, arm int, dollars float64) {
	if m == nil || arm == campaigns.ArmNone {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur != nil && m.cur.ID == id && m.cur.Status == "running" {
		m.cur.Arms[arm].Spend += dollars
	}
}

// ObserveShadow records the challenger's shadow prediction next to the
// champion's decision for the same request.
func (m *Manager) ObserveShadow(l *Live, placement string, d *policy.Decision, s *policy.ShadowScore, tsMS int64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e := m.cur
	if e == nil || e.ID != l.ID || e.Shadow == nil {
		return
	}
	e.clock(tsMS)
	st := e.Shadow
	st.N++
	st.clickC[placement] += d.PClick
	st.clickS[placement] += s.PClick
	if d.PInstallClick > 0 && s.PInstallClick > 0 {
		st.sumC[placement] += d.PInstallClick
		st.sumS[placement] += s.PInstallClick
		st.nI[placement]++
	}
	if len(st.lat) < 200000 {
		st.lat = append(st.lat, s.LatencyUS)
	}
}

func (m *Manager) checkCanary(nowMS int64) {
	e := m.cur
	c, t := e.Arms[campaigns.ArmControl], e.Arms[campaigns.ArmTreatment]
	var checks []Check
	keys := make([]string, 0, len(t.Paired))
	for k := range t.Paired {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		p := t.Paired[k]
		add := func(what string, n int64, canary, champ float64) {
			if n < MinSegmentN || canary <= 0 || champ <= 0 {
				return
			}
			r := math.Log(canary / champ)
			checks = append(checks, Check{Name: what + " ratio, " + k, Passed: math.Abs(r) <= MaxLogRatio,
				Value: r, Limit: MaxLogRatio, Detail: fmt.Sprintf("mean %.4f for the canary vs. %.4f for "+
					"the champion, on the same %d canary requests", canary/float64(n), champ/float64(n), n)})
		}
		add("click", p.N, p.Canary, p.Champion)
		add("install", p.NInst, p.InstC, p.InstH)
	}
	if c.All.N >= MinSegmentN && t.All.N >= MinSegmentN {
		fc, ft := float64(c.All.Fallback)/float64(c.All.N), float64(t.All.Fallback)/float64(t.All.N)
		checks = append(checks, Check{Name: "fallback share", Passed: ft-fc <= MaxFallbackPP,
			Value: ft - fc, Limit: MaxFallbackPP,
			Detail: fmt.Sprintf("%.1f%% of canary decisions vs. %.1f%% of control", 100*ft, 100*fc)})
	}
	if tot := c.Spend + t.Spend; tot >= MinSpend && e.TreatShare > 0 {
		r := (t.Spend / tot) / e.TreatShare
		checks = append(checks, Check{Name: "spend rate", Passed: r <= MaxSpendRatio, Value: r,
			Limit: MaxSpendRatio, Detail: fmt.Sprintf("canary spent $%.2f of $%.2f on %.0f%% of devices",
				t.Spend, tot, 100*e.TreatShare)})
	}
	e.Checks = checks
	for _, ch := range checks {
		if !ch.Passed {
			m.end(nowMS, "rolled back", fmt.Sprintf("%s %.3f outside the limit %.3g at stage %d (%.0f%%): %s",
				ch.Name, ch.Value, ch.Limit, e.Stage+1, 100*e.TreatShare, ch.Detail))
			return
		}
	}
	stageDone := float64(nowMS-e.StageStartMS) >= e.Config.StageHours*3600*1000
	if !stageDone || t.All.N < MinStageN || len(checks) == 0 {
		return
	}
	e.event(nowMS, fmt.Sprintf("stage %d (%.0f%%) passed %d checks", e.Stage+1, 100*e.TreatShare, len(checks)))
	if e.Stage+1 >= len(e.Config.Stages) {
		res := m.Registry.Deploy(e.Config.Challenger, e.Config.Force)
		if !res.OK {
			m.end(nowMS, "rolled back", "promotion failed: "+res.Message)
			return
		}
		m.end(nowMS, "promoted", fmt.Sprintf("every stage passed; %s is the champion", e.Config.Challenger))
		return
	}
	e.Stage++
	e.TreatShare = e.Config.Stages[e.Stage]
	e.StageStartMS = nowMS
	e.Arms = [2]*ArmStats{newArm(), newArm()} // each stage is judged on its own traffic
	e.event(nowMS, fmt.Sprintf("stage %d: %.0f%% of devices", e.Stage+1, 100*e.TreatShare))
	m.publish(nil, nil)
	m.save()
}

// Current returns a copy of the running or last experiment, or nil.
func (m *Manager) Current() *Experiment {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cur == nil {
		return nil
	}
	m.cur.summarizeShadow()
	raw, _ := json.Marshal(m.cur)
	var c Experiment
	json.Unmarshal(raw, &c)
	return &c
}

// List returns past experiments' records, newest first.
func (m *Manager) List() []json.RawMessage {
	ents, _ := os.ReadDir(m.Dir)
	out := []json.RawMessage{}
	for i := len(ents) - 1; i >= 0 && len(out) < 20; i-- {
		raw, err := os.ReadFile(filepath.Join(m.Dir, ents[i].Name()))
		if err == nil && json.Valid(raw) {
			out = append(out, raw)
		}
	}
	return out
}

func (e *Experiment) summarizeShadow() {
	s := e.Shadow
	if s == nil || s.sumC == nil {
		return
	}
	s.Champion, s.Challenger, s.ClickRatio = map[string]float64{}, map[string]float64{}, map[string]float64{}
	for p, n := range s.nI {
		s.Champion[p] = s.sumC[p] / n
		s.Challenger[p] = s.sumS[p] / n
	}
	for p, v := range s.clickC {
		if v > 0 {
			s.ClickRatio[p] = s.clickS[p] / v
		}
	}
	if len(s.lat) > 0 {
		l := append([]float64(nil), s.lat...)
		sort.Float64s(l)
		s.P50US, s.P99US = l[len(l)/2], l[int(0.99*float64(len(l)-1))]
	}
}

// clock starts an experiment's simulated clock at its first request, when
// it was started before any traffic arrived (a fresh bidder knows no time).
func (e *Experiment) clock(tsMS int64) {
	if e.StartedMS > 0 || tsMS <= 0 {
		return
	}
	e.StartedMS, e.StageStartMS, e.lastCheckMS = tsMS, tsMS, tsMS
	for i := range e.Events {
		if e.Events[i].TsMS == 0 {
			e.Events[i].TsMS = tsMS
		}
	}
}

func (e *Experiment) event(ts int64, what string) {
	e.Events = append(e.Events, Event{ts, what})
}

func (m *Manager) note(what string) {
	if m.Note != nil {
		m.Note(what)
	}
}

func (m *Manager) nextID() string {
	ents, _ := os.ReadDir(m.Dir)
	return fmt.Sprintf("exp-%04d", len(ents)+1)
}

func (m *Manager) save() {
	if m.Dir == "" || m.cur == nil {
		return
	}
	m.cur.summarizeShadow()
	os.MkdirAll(m.Dir, 0o755)
	raw, _ := json.MarshalIndent(m.cur, "", " ")
	tmp := filepath.Join(m.Dir, m.cur.ID+".json.tmp")
	if os.WriteFile(tmp, raw, 0o644) == nil {
		os.Rename(tmp, filepath.Join(m.Dir, m.cur.ID+".json"))
	}
}

// Interrupted marks an experiment record left running by a restart: the
// bidder doesn't resume experiments, so its devices go back to the
// champion.
func (m *Manager) Interrupted() {
	ents, _ := os.ReadDir(m.Dir)
	for _, en := range ents {
		p := filepath.Join(m.Dir, en.Name())
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		var e Experiment
		if json.Unmarshal(raw, &e) == nil && e.Status == "running" {
			e.Status, e.Reason = "interrupted", "the bidder restarted"
			out, _ := json.MarshalIndent(e, "", " ")
			os.WriteFile(p, out, 0o644)
		}
	}
}
