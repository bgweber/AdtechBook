package ops

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
)

// Alert is one of the few alerts that matter (Chapter 12). Each has an
// owner and a note on what to check first.
type Alert struct {
	Name       string  `json:"name"`
	Scope      string  `json:"scope"` // overall, a campaign, an exchange
	State      string  `json:"state"` // firing, resolved
	SinceMS    int64   `json:"since_ms"`
	ResolvedMS int64   `json:"resolved_ms,omitempty"`
	Value      float64 `json:"value"`
	Threshold  float64 `json:"threshold"`
	Detail     string  `json:"detail"`
	Owner      string  `json:"owner"`
	Check      string  `json:"check"`
}

// Thresholds, in one place so a reader can change them.
var (
	SpendRateMultiple         = 4.0   // spend in the last hour vs the plan for that hour
	SpendRateMinDollars       = 0.05  // ignore tiny campaigns at tiny scale
	SpendAheadOfPlan          = 1.5   // ... and only when well ahead of the day's plan
	SpendHourShare            = 0.10  // ... and the hour took at least this share of the day's budget
	SpendQuietHours     int64 = 2     // the first hours of a day have too little plan to judge
	NoBidShare                = 0.005 // bids per request below this is "no bids"
	FallbackShare             = 0.2   // share of model-era bids priced by the fallback
	CollapseRatio             = 1.5   // p99 / p50 of predictions below this
	JumpFactor                = 3.0   // p50 moving this much hour over hour
	LatencyP99MS              = 100.0
	ReconTolerance            = 0.02 // reconciliation gap per exchange and day
	MinRequests         int64 = 200
)

var owners = map[string][2]string{
	"spend_rate":          {"bidder on-call", "Campaign targets and bid multipliers changed in the audit log; pause the campaign if the cause isn't clear."},
	"no_bids":             {"bidder on-call", "Kill switches, the decision reasons on the Live tab, and the bidder's errors."},
	"fallback_spike":      {"ML team", "Whether the model is switched off, failed to load, or predicts outside its bounds."},
	"prediction_collapse": {"ML team", "The model deployed last; roll back by deploying the previous version, or switch the model off."},
	"latency":             {"bidder on-call", "The bidder's CPU and the latency card on the Live tab."},
	"reconciliation_gap":  {"bidder on-call", "Billing notices by exchange and placement: an exchange may have stopped sending them for one format."},
}

func (o *Ops) setAlertLocked(name, scope string, firing bool, ts int64, value, threshold float64, detail string) {
	k := name + "|" + scope
	a := o.alerts[k]
	if firing {
		if a != nil && a.State == "resolved" {
			o.history = append(o.history, *a) // keep the earlier episode
			if len(o.history) > 100 {
				o.history = o.history[len(o.history)-100:]
			}
		}
		if a == nil || a.State == "resolved" {
			a = &Alert{Name: name, Scope: scope, State: "firing", SinceMS: ts,
				Owner: owners[name][0], Check: owners[name][1]}
			o.alerts[k] = a
		}
		a.Value, a.Threshold, a.Detail = value, threshold, detail
		return
	}
	if a != nil && a.State == "firing" {
		a.State, a.ResolvedMS = "resolved", ts
	}
}

// evaluateLocked checks every alert on the last simulated hour.
func (o *Ops) evaluateLocked(ts int64, book *campaigns.Book) {
	hour := o.windowLocked(ts, 4)
	prev := o.windowLocked(ts-hourMS, 4)
	sw := o.sw.cur.Load()

	// Spend rate far above plan, by campaign and overall.
	spend := map[int]float64{}
	for _, q := range hour {
		for c, d := range q.spend {
			spend[c] += d
		}
	}
	// The plan for the last hour follows the day's traffic shape, and a
	// campaign catching up after falling behind isn't a runaway: it must
	// also be ahead of its plan for the day so far.
	day := int(ts / clock.DayMS)
	end := ts - ts%QuarterMS
	hourShare := o.plannedShareLocked(end) - o.plannedShareLocked(end-hourMS)
	if end%clock.DayMS < hourMS { // the hour crosses midnight
		hourShare = o.plannedShareLocked(end)
	}
	soFar := o.plannedShareLocked(end)
	sinceMidnight := end % clock.DayMS
	totalSpend, totalPlan, totalToday, totalBudget := 0.0, 0.0, 0.0, 0.0
	for _, v := range book.Views(day) {
		plan := v.DailyBudget * hourShare
		s := spend[v.ID]
		ahead := v.SpendToday > SpendAheadOfPlan*v.DailyBudget*soFar
		totalSpend += s
		totalPlan += plan
		totalToday += v.SpendToday
		totalBudget += v.DailyBudget
		o.setAlertLocked("spend_rate", "campaign "+strconv.Itoa(v.ID),
			s > SpendRateMultiple*plan && s > SpendRateMinDollars && s > SpendHourShare*v.DailyBudget &&
				ahead && sinceMidnight >= SpendQuietHours*hourMS,
			ts, s, SpendRateMultiple*plan,
			fmt.Sprintf("$%.2f in the last simulated hour, against a plan of $%.2f; $%.2f of its $%.2f budget spent today, %.0f%% of the day's plan so far",
				s, plan, v.SpendToday, v.DailyBudget, 100*v.SpendToday/math.Max(v.DailyBudget*soFar, 1e-9)))
	}
	o.setAlertLocked("spend_rate", "overall", totalSpend > SpendRateMultiple*totalPlan &&
		totalSpend > SpendRateMinDollars && totalSpend > SpendHourShare*totalBudget &&
		totalToday > SpendAheadOfPlan*totalBudget*soFar && sinceMidnight >= SpendQuietHours*hourMS,
		ts, totalSpend, SpendRateMultiple*totalPlan,
		fmt.Sprintf("$%.2f in the last simulated hour, against a plan of $%.2f for all campaigns", totalSpend, totalPlan))

	// No bids while requests keep arriving, by exchange and overall.
	req, bids := map[string]int64{}, map[string]int64{}
	for _, q := range hour {
		for e, n := range q.requests {
			req[e] += n
			req["overall"] += n
		}
		for e, n := range q.bids {
			bids[e] += n
			bids["overall"] += n
		}
	}
	for e, n := range req {
		share := float64(bids[e]) / float64(max(n, 1))
		why := ""
		if sw.Global {
			why = " The global kill switch is on."
		} else if sw.Exchanges[e] {
			why = " This exchange's kill switch is on."
		}
		scope := e
		if e != "overall" {
			scope = "exchange " + e
		}
		if n < MinRequests {
			continue // too little traffic to judge: keep the alert as it is
		}
		o.setAlertLocked("no_bids", scope, !o.listen && share < NoBidShare, ts, share, NoBidShare,
			fmt.Sprintf("%d bids on %d requests in the last simulated hour.%s", bids[e], n, why))
	}

	// Fallback use spikes while a model is deployed.
	var priced, fb int64
	var pred, pprev [predBins]int64
	for _, q := range hour {
		priced += q.priced
		fb += q.fallback
		for i, c := range q.pred {
			pred[i] += c
		}
	}
	for _, q := range prev {
		for i, c := range q.pred {
			pprev[i] += c
		}
	}
	share := float64(fb) / float64(max(priced, 1))
	why := ""
	if sw.Model {
		why = " The model's kill switch is on."
	}
	if priced >= 100 || !o.modelOn {
		o.setAlertLocked("fallback_spike", "click model", o.modelOn && share > FallbackShare, ts,
			share, FallbackShare, fmt.Sprintf("%s of bids in the last simulated hour were priced by the fallback.%s",
				fmtPct(share), why))
	}

	// The prediction distribution collapses or jumps.
	p50, n := quantile(pred, 0.5)
	p99, _ := quantile(pred, 0.99)
	q50, m := quantile(pprev, 0.5)
	collapsed := n >= MinRequests && p99/math.Max(p50, 1e-9) < CollapseRatio
	jumped := n >= MinRequests && m >= MinRequests &&
		(p50 > JumpFactor*q50 || p50 < q50/JumpFactor)
	detail := fmt.Sprintf("Click predictions in the last simulated hour: p50 %s, p99 %s", fmtPct(p50), fmtPct(p99))
	if m >= MinRequests {
		detail += fmt.Sprintf("; p50 the hour before %s", fmtPct(q50))
	}
	if n >= MinRequests || !o.modelOn {
		o.setAlertLocked("prediction_collapse", "click model", collapsed || jumped, ts,
			p99/math.Max(p50, 1e-9), CollapseRatio, detail+".")
	}

	// Latency, inside the bidder.
	var lat [32]int64
	var ln int64
	for _, q := range hour {
		for i, c := range q.latency {
			lat[i] += c
		}
		ln += q.latencyN
	}
	if ln >= MinRequests {
		target, seen, p := int64(math.Ceil(0.99*float64(ln))), int64(0), 0.0
		for i, c := range lat {
			seen += c
			if seen >= target {
				p = math.Pow(2, float64(i+1)) / 1000
				break
			}
		}
		o.setAlertLocked("latency", "overall", p > LatencyP99MS, ts, p, LatencyP99MS,
			fmt.Sprintf("p99 latency inside the bidder up to %.0f ms in the last simulated hour.", p))
	}

	o.evaluateReconLocked(ts)
}

// Alerts returns the alerts, firing first, then newest first.
func (o *Ops) Alerts() []Alert {
	o.mu.Lock()
	defer o.mu.Unlock()
	out := make([]Alert, 0, len(o.alerts)+len(o.history))
	for _, a := range o.alerts {
		out = append(out, *a)
	}
	out = append(out, o.history...)
	sort.Slice(out, func(i, j int) bool {
		if (out[i].State == "firing") != (out[j].State == "firing") {
			return out[i].State == "firing"
		}
		return out[i].SinceMS > out[j].SinceMS
	})
	return out
}
