package ops

import (
	"fmt"
	"sort"
)

// Incident drills (Chapter 24): faults injected into a running bidder on
// purpose, so the team can practice following an alert to its playbook.
// Each is logged in the audit log like any other change, which is how the
// alert finds it.
//
//   - shift: a deploy whose click predictions are too high everywhere. It
//     belongs to the conversion bundle that's serving when it starts, so a
//     rollback to any other bundle ends it, as rolling back a bad build does.
//   - parity: the serving code for one feature changes: connection types
//     become finer than training expects (the Chapter 8 skew).
//
// The third drill, a day of bad install labels, comes from the simulator:
// ADSIM_SCENARIO=bad-labels.
type Drill struct {
	Kind      string  `json:"kind"`
	Version   string  `json:"version,omitempty"` // the bundle a shift belongs to
	Factor    float64 `json:"factor,omitempty"`
	Feature   string  `json:"feature,omitempty"`
	StartedMS int64   `json:"started_ms"`
}

// StartShiftDrill multiplies the click predictions of a bundle version.
func (o *Ops) StartShiftDrill(version string, factor float64) {
	o.mu.Lock()
	if o.drills == nil {
		o.drills = map[string]Drill{}
	}
	o.drills["shift"] = Drill{Kind: "shift", Version: version, Factor: factor, StartedMS: o.nowMS}
	o.shiftVersion.Store(&shiftFault{version, factor})
	o.mu.Unlock()
	o.Note(fmt.Sprintf("conversion bundle %s deployed (drill: click predictions x%.2f)", version, factor),
		"drill", 0)
}

// StartParityDrill records the parity drill; the caller changes the
// feature code.
func (o *Ops) StartParityDrill(feature string) {
	o.mu.Lock()
	if o.drills == nil {
		o.drills = map[string]Drill{}
	}
	o.drills["parity"] = Drill{Kind: "parity", Feature: feature, StartedMS: o.nowMS}
	o.mu.Unlock()
	o.Note("bidder deployed (drill: serving code for "+feature+" changed)", "drill", 0)
}

// StopDrill ends a drill.
func (o *Ops) StopDrill(kind string) {
	o.mu.Lock()
	_, ok := o.drills[kind]
	delete(o.drills, kind)
	if kind == "shift" {
		o.shiftVersion.Store(nil)
	}
	o.mu.Unlock()
	if ok {
		o.Note("drill ended: "+kind, "drill", 0)
	}
}

// Drills lists the running drills.
func (o *Ops) Drills() []Drill {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := []Drill{}
	for _, d := range o.drills {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Kind < out[j].Kind })
	return out
}

type shiftFault struct {
	version string
	factor  float64
}

// Shift is the drill's multiplier on a bundle's click predictions: 1
// unless the shift drill is running on that version. It's on the bid path,
// so it takes no lock.
func (o *Ops) Shift(version string) float64 {
	if f := o.shiftVersion.Load(); f != nil && f.version == version {
		return f.factor
	}
	return 1
}
