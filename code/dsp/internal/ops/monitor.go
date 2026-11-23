package ops

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// Prediction monitoring against baselines (Chapter 24). Every simulated
// quarter hour, the mean prediction of each model over the last hour is
// compared with the same hour on the days before, the baseline:
//
//	delta = ln(mean now / median of the baseline days' means)
//	z     = delta / (1.4826 x MAD of the baseline days, in log space)
//
// A shift alert fires when |z| is above ShiftZ, and |delta| above
// ShiftMinLog, for ShiftIntervals quarter hours in a row. The population
// stability index of the hour's predictions against the baseline days'
// is shown with it. Prediction distributions cycle through the day, so a
// fixed threshold would fire every evening; the same hour on earlier days
// moves with the cycle.
var (
	ShiftZ         = 4.0
	ShiftMinLog    = 0.05 // about a 5% change in the mean
	ShiftIntervals = 3    // consecutive quarter hours
	ShiftBaseDays  = 4    // the same hour on up to this many days before
	ShiftMinN      = int64(200)
	shiftMADFloor  = 0.02 // in log space: baselines that happen to agree closely don't make every wobble a page
)

// Monitored are the models' outputs the monitor follows: the click model
// and the install-after-click model.
var Monitored = []string{"click", "install"}

const monitorKeepQuarters = 6 * 96 // six simulated days

type predStat struct {
	N    int64           `json:"n"`
	Sum  float64         `json:"sum"`
	Hist [predBins]int64 `json:"hist"`
}

func (s *predStat) observe(p float64) {
	s.N++
	s.Sum += p
	s.Hist[predBin(p)]++
}

func (s *predStat) merge(o *predStat) {
	if o == nil {
		return
	}
	s.N += o.N
	s.Sum += o.Sum
	for i, c := range o.Hist {
		s.Hist[i] += c
	}
}

func (s *predStat) mean() float64 { return s.Sum / math.Max(1, float64(s.N)) }

// ShiftPoint is one evaluation of one model's output, for the console's
// chart and the alert.
type ShiftPoint struct {
	TsMS     int64   `json:"ts_ms"`
	Target   string  `json:"target"`
	N        int64   `json:"n"`
	Mean     float64 `json:"mean"`
	Base     float64 `json:"base"` // median of the baseline days' means
	BaseLo   float64 `json:"base_lo"`
	BaseHi   float64 `json:"base_hi"`
	BaseDays int     `json:"base_days"`
	Delta    float64 `json:"delta"` // ln(mean / base)
	Z        float64 `json:"z"`
	PSI      float64 `json:"psi"`
	Over     bool    `json:"over"` // beyond the thresholds this quarter
}

type monitor struct {
	Q      map[int64]map[string]*predStat `json:"quarters"`
	Points []ShiftPoint                   `json:"points"`
	streak map[string]int
	calm   map[string]int
	began  map[string]int64 // when the current run of shifted quarters began
}

func (o *Ops) mon() *monitor {
	if o.monitor == nil {
		o.monitor = &monitor{Q: map[int64]map[string]*predStat{}}
	}
	if o.monitor.streak == nil {
		o.monitor.streak, o.monitor.calm, o.monitor.began = map[string]int{}, map[string]int{}, map[string]int64{}
	}
	return o.monitor
}

// observePredLocked counts one priced bid's predictions.
func (o *Ops) observePredLocked(ts int64, target string, p float64) {
	if p <= 0 || math.IsNaN(p) {
		return
	}
	m := o.mon()
	k := ts / QuarterMS
	byT := m.Q[k]
	if byT == nil {
		byT = map[string]*predStat{}
		m.Q[k] = byT
		for old := range m.Q {
			if old < k-monitorKeepQuarters {
				delete(m.Q, old)
			}
		}
	}
	s := byT[target]
	if s == nil {
		s = &predStat{}
		byT[target] = s
	}
	s.observe(p)
}

// windowStat sums one target over the hour of quarters ending before k.
func (m *monitor) windowStat(k int64, target string) *predStat {
	s := &predStat{}
	for i := k - 4; i < k; i++ {
		s.merge(m.Q[i][target])
	}
	return s
}

// evaluateShiftLocked runs the shift check for every monitored model.
func (o *Ops) evaluateShiftLocked(ts int64) {
	m := o.mon()
	k := ts / QuarterMS
	for _, t := range Monitored {
		cur := m.windowStat(k, t)
		if cur.N < ShiftMinN {
			continue // too few predictions to judge: keep the alert as it is
		}
		var base []float64
		pooled := &predStat{}
		for d := int64(1); d <= int64(ShiftBaseDays); d++ {
			b := m.windowStat(k-96*d, t)
			if b.N >= ShiftMinN {
				base = append(base, math.Log(b.mean()))
				pooled.merge(b)
			}
		}
		pt := ShiftPoint{TsMS: ts - ts%QuarterMS, Target: t, N: cur.N, Mean: round6(cur.mean()),
			BaseDays: len(base)}
		if len(base) >= 2 {
			med := median(base)
			dev := make([]float64, len(base))
			for i, b := range base {
				dev[i] = math.Abs(b - med)
			}
			scale := math.Max(1.4826*median(dev), shiftMADFloor)
			pt.Base = round6(math.Exp(med))
			pt.BaseLo, pt.BaseHi = round6(math.Exp(minOf(base))), round6(math.Exp(maxOf(base)))
			pt.Delta = math.Round((math.Log(cur.mean())-med)*1e4) / 1e4
			pt.Z = math.Round(pt.Delta/scale*100) / 100
			pt.PSI = math.Round(psi(cur, pooled)*1e4) / 1e4
			pt.Over = math.Abs(pt.Z) > ShiftZ && math.Abs(pt.Delta) > ShiftMinLog
		}
		m.Points = append(m.Points, pt)
		if pt.Over {
			if m.streak[t] == 0 {
				m.began[t] = ts
			}
			m.streak[t]++
			m.calm[t] = 0
		} else {
			m.calm[t]++
			m.streak[t] = 0
		}
		scope := t + " model"
		a := o.alerts["prediction_shift|"+scope]
		firing := a != nil && a.State == "firing"
		switch {
		case m.streak[t] >= ShiftIntervals:
			firing = true
		case m.calm[t] >= 2: // two calm quarters in a row before it resolves
			firing = false
		}
		if len(base) < 2 && !firing {
			continue
		}
		dir := "above"
		if pt.Delta < 0 {
			dir = "below"
		}
		detail := fmt.Sprintf("Mean %s prediction over the last simulated hour %s, %.0f%% %s the same hour "+
			"on the last %d days (%s); z %.1f, PSI %.3f.", t, fmtPct(pt.Mean),
			100*math.Abs(math.Exp(pt.Delta)-1), dir, pt.BaseDays, fmtPct(pt.Base), pt.Z, pt.PSI)
		if dep := o.lastDeploy(m.began[t]); dep != "" {
			detail += " " + dep
		}
		o.setAlertLocked("prediction_shift", scope, firing, ts, pt.Z, ShiftZ, detail)
	}
	if len(m.Points) > 2*96*len(Monitored) {
		m.Points = m.Points[len(m.Points)-2*96*len(Monitored):]
	}
	if k%4 == 0 {
		o.saveMonitorLocked()
	}
}

// lastDeploy annotates an alert with the newest model or bidder change in
// the audit log in the six simulated hours before the shift began
// (Chapter 24: most real-time alerts follow a deploy).
func (o *Ops) lastDeploy(ts int64) string { return o.lastChange(ts, 6*hourMS) }

// lastChange is the newest change in the audit log in the window before ts.
func (o *Ops) lastChange(ts, window int64) string {
	for _, e := range o.Audit() {
		w := strings.ToLower(e.What)
		if !strings.Contains(w, "deploy") && !strings.Contains(w, "promot") &&
			!strings.Contains(w, "roll") && !strings.Contains(w, "drill") {
			continue
		}
		if e.SimMS > ts {
			continue // after the shift began
		}
		if e.SimMS > 0 && ts-e.SimMS <= window {
			return fmt.Sprintf("Last change before it: %s, at %s.", e.What, simClock(e.SimMS))
		}
		return ""
	}
	return ""
}

func simClock(ms int64) string {
	m := (ms % clock.DayMS) / 60_000
	return fmt.Sprintf("day %d, %02d:%02d", ms/clock.DayMS, m/60, m%60)
}

// psi is the population stability index of a window's predictions
// against the baseline's, in 16 bins on the log scale.
func psi(cur, base *predStat) float64 {
	const groups = 16
	var a, e [groups]float64
	for i := 0; i < predBins; i++ {
		a[i*groups/predBins] += float64(cur.Hist[i])
		e[i*groups/predBins] += float64(base.Hist[i])
	}
	out := 0.0
	for i := range a {
		ai := math.Max(a[i]/math.Max(1, float64(cur.N)), 1e-4)
		ei := math.Max(e[i]/math.Max(1, float64(base.N)), 1e-4)
		out += (ai - ei) * math.Log(ai/ei)
	}
	return out
}

func median(xs []float64) float64 {
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

func minOf(xs []float64) float64 {
	m := math.Inf(1)
	for _, x := range xs {
		m = math.Min(m, x)
	}
	return m
}

func maxOf(xs []float64) float64 {
	m := math.Inf(-1)
	for _, x := range xs {
		m = math.Max(m, x)
	}
	return m
}

func round6(x float64) float64 { return math.Round(x*1e6) / 1e6 }

// ShiftView is the monitor's recent evaluations, for the console.
func (o *Ops) ShiftView() []ShiftPoint {
	o.mu.Lock()
	defer o.mu.Unlock()
	return append([]ShiftPoint(nil), o.mon().Points...)
}

// The monitor's counts are saved, so a restarted bidder keeps its
// baselines.
func (o *Ops) saveMonitorLocked() {
	if o.sw.dir == "" || o.monitor == nil {
		return
	}
	os.MkdirAll(o.sw.dir, 0o755)
	raw, err := json.Marshal(o.monitor)
	if err != nil {
		return
	}
	tmp := filepath.Join(o.sw.dir, "monitor.json.tmp")
	if os.WriteFile(tmp, raw, 0o644) == nil {
		os.Rename(tmp, filepath.Join(o.sw.dir, "monitor.json"))
	}
}

func (o *Ops) loadMonitor() {
	raw, err := os.ReadFile(filepath.Join(o.sw.dir, "monitor.json"))
	if err != nil {
		return
	}
	var m monitor
	if json.Unmarshal(raw, &m) == nil && m.Q != nil {
		o.monitor = &m
	}
}
