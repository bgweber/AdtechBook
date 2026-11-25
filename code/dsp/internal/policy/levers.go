package policy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync/atomic"
)

// Levers are the per-campaign settings of the unified pricing equation
// (Chapter 33) that the campaign book doesn't carry: a second goal with its
// weight, and the strength of the incrementality tilt. The account manager
// sets them on the console; every one acts on one term of the equation.
type Levers struct {
	// A second goal, CPI, weighted against the campaign's own: the value is
	// (1 - Weight) x its goal's value + Weight x the CPI goal's value, each
	// with the campaign's calibration multiplier.
	SecondGoal   string  `json:"second_goal,omitempty"`
	SecondTarget float64 `json:"second_target,omitempty"`
	Weight       float64 `json:"weight,omitempty"`
	// TiltAlpha is the incrementality tilt's strength, 0 to 1, with the
	// uplift bundle the levers name (Chapter 32).
	TiltAlpha float64 `json:"tilt_alpha,omitempty"`
}

// LeverSet is every campaign's levers, and the uplift bundle the tilts use.
type LeverSet struct {
	Campaigns map[int]Levers `json:"campaigns"`
	Uplift    string         `json:"uplift,omitempty"`
}

// Check says what's wrong with a campaign's levers, if anything.
func (l Levers) Check() error {
	if l.SecondGoal != "" && l.SecondGoal != "CPI" {
		return fmt.Errorf("the second goal can be CPI only")
	}
	if l.SecondGoal != "" && (l.SecondTarget <= 0 || l.Weight <= 0 || l.Weight >= 1) {
		return fmt.Errorf("a second goal needs a target above 0 and a weight between 0 and 1")
	}
	if l.TiltAlpha < 0 || l.TiltAlpha > 1 {
		return fmt.Errorf("the tilt's alpha is between 0 and 1")
	}
	return nil
}

// levers holds the current set, read once per priced campaign without a lock.
type levers struct{ p atomic.Pointer[LeverSet] }

// Levers returns the current set (never nil).
func (p *Crawl) Levers() *LeverSet {
	if s := p.levers.p.Load(); s != nil {
		return s
	}
	return &LeverSet{Campaigns: map[int]Levers{}}
}

// SetLevers swaps in a new set, and saves it to path when path isn't "".
func (p *Crawl) SetLevers(s *LeverSet, path string) error {
	if s.Campaigns == nil {
		s.Campaigns = map[int]Levers{}
	}
	for id, l := range s.Campaigns {
		if err := l.Check(); err != nil {
			return fmt.Errorf("campaign %d: %w", id, err)
		}
	}
	if path != "" {
		raw, _ := json.MarshalIndent(s, "", " ")
		os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, raw, 0o644); err != nil {
			return err
		}
	}
	p.levers.p.Store(s)
	return nil
}

// LoadLevers reads a saved set; a missing file is an empty set.
func (p *Crawl) LoadLevers(path string) error {
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var s LeverSet
	if err := json.Unmarshal(raw, &s); err != nil {
		return err
	}
	return p.SetLevers(&s, "")
}

// GoalTerm is one goal's part of a weighted value, for the bid ledger.
type GoalTerm struct {
	Goal   string  `json:"goal"`
	Target float64 `json:"target"`
	Weight float64 `json:"weight"`
	Value  float64 `json:"value"` // the advertiser value under this goal, CPM, before calibration
	Kappa  float64 `json:"kappa"`
}
