package console

import (
	"encoding/json"
	"fmt"
	"log"
	"sort"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/ops"
)

// StartDaily runs the console's daily jobs (Chapter 24): when the
// simulated day turns, the training service checks the day that just
// ended for feature parity, and the result raises or clears the parity
// alert. Parity checks run as a daily batch job in production too.
func StartDaily(d Deps) {
	if d.Trainer == nil || d.Trainer.URL == "" {
		return
	}
	go func() {
		last := int64(-1)
		for {
			time.Sleep(5 * time.Second)
			day := d.Stats.Snapshot().SimDay
			if day < 0 {
				continue
			}
			if last < 0 {
				last = day
				continue
			}
			if day > last {
				last = day
				r, err := checkParity(d, day-1)
				if err != nil {
					log.Printf("daily parity check of day %d: %v", day-1, err)
					continue
				}
				log.Printf("daily parity check of day %d: %d rows, worst feature %s at %.2f%%",
					r.Day, r.Rows, r.Feature, 100*r.Rate)
			}
		}
	}()
}

// checkParity runs the parity check on one day and records the result.
func checkParity(d Deps, day int64) (*ops.ParityResult, error) {
	raw, err := d.Trainer.ReportWith("parity", map[string]any{"days": []int64{day}})
	if err != nil {
		return nil, err
	}
	var rep struct {
		Rows     int64 `json:"rows"`
		Features []struct {
			Feature    string  `json:"feature"`
			Rate       float64 `json:"rate"`
			Mismatched int64   `json:"mismatched"`
			Examples   []struct {
				Exchange   string          `json:"exchange"`
				Logged     json.RawMessage `json:"logged"`
				Recomputed json.RawMessage `json:"recomputed"`
			} `json:"examples"`
		} `json:"features"`
	}
	if err := json.Unmarshal(raw, &rep); err != nil {
		return nil, err
	}
	res := ops.ParityResult{Day: day, Rows: rep.Rows}
	sort.SliceStable(rep.Features, func(i, j int) bool { return rep.Features[i].Rate > rep.Features[j].Rate })
	if len(rep.Features) > 0 {
		f := rep.Features[0]
		res.Feature, res.Rate = f.Feature, f.Rate
		if len(f.Examples) > 0 && f.Mismatched > 0 {
			e := f.Examples[0]
			res.Examples = fmt.Sprintf("logged %s, recomputed %s on %s", e.Logged, e.Recomputed, e.Exchange)
		}
	}
	d.Ops.SetParity(res)
	return &res, nil
}
