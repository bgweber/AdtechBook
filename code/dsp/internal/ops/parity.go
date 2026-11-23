package ops

import (
	"fmt"

	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// ParityThreshold is the most a feature's mismatch rate can be in the
// daily parity check before it alerts (Chapter 24: below 0.1% per feature).
var ParityThreshold = 0.001

// ParityResult is what the daily parity job reports for one day.
type ParityResult struct {
	Day      int64   `json:"day"`
	Rows     int64   `json:"rows"`
	Feature  string  `json:"feature"` // the feature with the most mismatches
	Rate     float64 `json:"rate"`
	Examples string  `json:"examples,omitempty"`
}

// SetParity records the daily parity check's result and raises or clears
// the parity alert. Too few rows to judge keeps the alert as it is.
func (o *Ops) SetParity(r ParityResult) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.parity = &r
	if r.Rows < 100 {
		return
	}
	detail := fmt.Sprintf("Day %d's parity check: %s of %d rows have a different %s when the training code "+
		"recomputes it from the raw request (limit %s).", r.Day, fmtPct(r.Rate), r.Rows, r.Feature,
		fmtPct(ParityThreshold))
	if r.Examples != "" {
		detail += " For example: " + r.Examples + "."
	}
	if dep := o.lastChange((r.Day+1)*clock.DayMS, clock.DayMS); dep != "" && r.Rate > ParityThreshold {
		detail += " " + dep
	}
	o.setAlertLocked("parity", "features", r.Rate > ParityThreshold, o.nowMS, r.Rate, ParityThreshold, detail)
}

// LastParity is the newest daily parity result, or nil.
func (o *Ops) LastParity() *ParityResult {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.parity
}
