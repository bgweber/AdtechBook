package ops

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// Pacing by bid modification (Chapter 22). Every simulated quarter hour, a
// PI controller compares each campaign's spend so far today with its plan
// and sets its pacing multiplier gamma, which scales the campaign's value
// before the margin and shading:
//
//	e = ln((plan + d) / (spend + d))
//	ln gamma = ln gamma0 + Kp e + Ki sum(e)
//
// with d a small share of the budget, so the first impressions of a day
// don't swing gamma. The plan is shaped like yesterday's requests, as the
// throttling plan is. gamma0 is the replay forecast for the day, if one
// was loaded, or the last gamma of the day before. Throttling stays as a
// backstop: a campaign far ahead of plan even at the lowest gamma skips
// requests.
const (
	PaceKp       = 0.5 // about 1 / spend elasticity, which is 1.2-2 here
	PaceKi       = 0.05
	PaceMin      = 0.3
	PaceMax      = 2.0
	PaceMaxStep  = 0.2  // most ln gamma moves in one step
	paceSlack    = 0.02 // d, as a share of the daily budget
	paceBackstop = 1.3  // spend over plan, at the lowest gamma, before throttling
	// PaceRampStart caps gamma when a day starts (Chapter 24): the
	// controller holds still until 1% of the day's plan has passed, so a
	// campaign that started at its gamma0 of 2 bid twice its value through
	// the quiet first hour, while its plan was near zero. Starting at no
	// more than 1, the step limit raises it toward gamma0 over the first
	// hour of control instead.
	PaceRampStart = 1.0
)

// PaceState is one campaign's controller.
type PaceState struct {
	Gamma    float64     `json:"gamma"`
	Gamma0   float64     `json:"gamma0"`
	From     string      `json:"gamma0_from"` // forecast, yesterday, or default
	Integral float64     `json:"integral"`
	Overbid  float64     `json:"overbid,omitempty"` // manual, from the console; 1 is none
	Today    []PacePoint `json:"today"`
	Day      int64       `json:"day"`
}

// PacePoint is one quarter hour of a campaign's day, for the console.
type PacePoint struct {
	TsMS  int64   `json:"ts_ms"`
	Plan  float64 `json:"plan"`  // planned spend so far today, dollars
	Spend float64 `json:"spend"` // actual
	Gamma float64 `json:"gamma"`
}

// SetPacing switches between throttling (off, Part II) and the controller.
func (o *Ops) SetPacing(on bool) {
	o.mu.Lock()
	o.pacing = on
	o.savePacingLocked()
	o.mu.Unlock()
}

// Pacing reports whether the controller is on.
func (o *Ops) Pacing() bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.pacing
}

// Pace is a campaign's multipliers for pricing: gamma (1 when the
// controller is off) and its manual overbid (1 when none).
func (o *Ops) Pace(campaignID int) (gamma, overbid float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	gamma, overbid = 1, 1
	if s := o.pace[campaignID]; s != nil {
		if o.pacing && s.Gamma > 0 {
			gamma = s.Gamma
		}
		if s.Overbid > 0 {
			overbid = s.Overbid
		}
	}
	return
}

// SetOverbid sets a campaign's manual overbid: a fixed multiplier on its
// value that account managers used before pacing (Chapter 22). 1 removes it.
func (o *Ops) SetOverbid(campaignID int, m float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	s := o.paceStateLocked(campaignID)
	s.Overbid = math.Max(0.1, math.Min(5, m))
	o.savePacingLocked()
}

// SetForecast loads each campaign's starting gamma for a day, from the
// replay forecast. The controller starts the day there instead of at
// yesterday's last value.
func (o *Ops) SetForecast(day int64, gamma0 map[int]float64) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.forecast, o.forecastDay = gamma0, day
	o.savePacingLocked()
}

// PaceView is every campaign's controller, for the console.
func (o *Ops) PaceView() (bool, map[int]*PaceState) {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := map[int]*PaceState{}
	for k, v := range o.pace {
		c := *v
		c.Today = append([]PacePoint(nil), v.Today...)
		out[k] = &c
	}
	return o.pacing, out
}

func (o *Ops) paceStateLocked(id int) *PaceState {
	if o.pace == nil {
		o.pace = map[int]*PaceState{}
	}
	s := o.pace[id]
	if s == nil {
		s = &PaceState{Gamma: 1, Gamma0: 1, From: "default", Day: -1}
		o.pace[id] = s
	}
	return s
}

// updatePacingLocked runs one step of every campaign's controller.
func (o *Ops) updatePacingLocked(ts int64, book *campaigns.Book) {
	if !o.pacing {
		return
	}
	day := ts / clock.DayMS
	plan := o.plannedShareLocked(ts)
	for _, v := range book.Views(int(day)) {
		if v.DailyBudget <= 0 || int64(v.StartDay) > day {
			continue
		}
		s := o.paceStateLocked(v.ID)
		if s.Day != day {
			// a new day: start from the forecast, or where yesterday ended
			s.Gamma0, s.From = s.Gamma, "yesterday"
			if s.Day < 0 {
				s.Gamma0, s.From = 1, "default"
			}
			if g, ok := o.forecast[v.ID]; ok && o.forecastDay == day && g > 0 {
				s.Gamma0, s.From = g, "forecast"
			}
			s.Gamma, s.Integral, s.Day, s.Today = math.Min(s.Gamma0, PaceRampStart), 0, day, nil
		}
		planned := v.DailyBudget * plan
		if plan >= 0.01 {
			d := paceSlack * v.DailyBudget
			e := math.Log((planned + d) / (v.SpendToday + d))
			s.Integral += e
			target := math.Log(s.Gamma0) + PaceKp*e + PaceKi*s.Integral
			cur := math.Log(s.Gamma)
			step := math.Max(-PaceMaxStep, math.Min(PaceMaxStep, target-cur))
			s.Gamma = math.Max(PaceMin, math.Min(PaceMax, math.Exp(cur+step)))
		}
		s.Today = append(s.Today, PacePoint{TsMS: ts, Plan: round2(planned), Spend: round2(v.SpendToday),
			Gamma: math.Round(s.Gamma*1000) / 1000})
		// the backstop: far ahead of plan even at the lowest gamma
		r := 1.0
		if s.Gamma <= PaceMin*1.001 && v.SpendToday > paceBackstop*planned && planned > 0 {
			r = 0.3
		}
		o.rates[v.ID] = r
	}
	o.savePacingLocked()
}

func round2(x float64) float64 { return math.Round(x*100) / 100 }

func (o *Ops) savePacingLocked() {
	if o.sw.dir == "" {
		return
	}
	os.MkdirAll(o.sw.dir, 0o755)
	raw, _ := json.Marshal(map[string]any{"on": o.pacing, "campaigns": o.pace,
		"forecast": o.forecast, "forecast_day": o.forecastDay})
	tmp := filepath.Join(o.sw.dir, "pacing.json.tmp")
	if os.WriteFile(tmp, raw, 0o644) == nil {
		os.Rename(tmp, filepath.Join(o.sw.dir, "pacing.json"))
	}
}

func (o *Ops) loadPacing() {
	raw, err := os.ReadFile(filepath.Join(o.sw.dir, "pacing.json"))
	if err != nil {
		return
	}
	var st struct {
		On       bool               `json:"on"`
		Camps    map[int]*PaceState `json:"campaigns"`
		Forecast map[int]float64    `json:"forecast"`
		FDay     int64              `json:"forecast_day"`
	}
	if json.Unmarshal(raw, &st) == nil {
		o.pacing, o.pace, o.forecast, o.forecastDay = st.On, st.Camps, st.Forecast, st.FDay
	}
}

// PaceIDs lists campaigns with controller state, sorted.
func PaceIDs(m map[int]*PaceState) []int {
	ids := make([]int, 0, len(m))
	for k := range m {
		ids = append(ids, k)
	}
	sort.Ints(ids)
	return ids
}
