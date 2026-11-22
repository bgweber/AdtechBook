package ops

import (
	"fmt"
	"math"
	"sort"
)

// Spend reconciliation (Chapter 12): our billed spend, counted from
// billing notices, against what each exchange invoices, by day,
// exchange, and placement.

type reconKey struct {
	Day       int64  `json:"day"`
	Exchange  string `json:"exchange"`
	Placement string `json:"placement"`
}

type reconCell struct {
	Imps  int64   `json:"impressions"`
	Spend float64 `json:"spend"`
}

// InvoiceLine is one line of an exchange's invoice.
type InvoiceLine struct {
	Day       int64   `json:"day"`
	Exchange  string  `json:"exchange"`
	Placement string  `json:"placement"`
	Imps      int64   `json:"impressions"`
	Spend     float64 `json:"spend"`
}

// LoadInvoices replaces the invoices for the days they cover.
func (o *Ops) LoadInvoices(lines []InvoiceLine) int {
	o.mu.Lock()
	defer o.mu.Unlock()
	days := map[int64]bool{}
	for _, l := range lines {
		days[l.Day] = true
	}
	for k := range o.invoiced {
		if days[k.Day] {
			delete(o.invoiced, k)
		}
	}
	for _, l := range lines {
		k := reconKey{l.Day, l.Exchange, l.Placement}
		c := o.invoiced[k]
		if c == nil {
			c = &reconCell{}
			o.invoiced[k] = c
		}
		c.Imps += l.Imps
		c.Spend += l.Spend
	}
	o.evaluateReconLocked(o.nowMS)
	return len(lines)
}

// ReconRow is one day and exchange, with its placements.
type ReconRow struct {
	Day        int64       `json:"day"`
	Exchange   string      `json:"exchange"`
	Ours       reconCell   `json:"ours"`
	Invoice    reconCell   `json:"invoice"`
	Gap        float64     `json:"gap"` // (ours - invoice) / invoice
	Placements []ReconLine `json:"placements"`
}

// ReconLine is one placement within a day and exchange.
type ReconLine struct {
	Placement string    `json:"placement"`
	Ours      reconCell `json:"ours"`
	Invoice   reconCell `json:"invoice"`
	Gap       float64   `json:"gap"`
}

// Reconciliation returns the days that have invoices, newest first.
func (o *Ops) Reconciliation() []ReconRow {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.reconLocked()
}

func (o *Ops) reconLocked() []ReconRow {
	type dk struct {
		day int64
		ex  string
	}
	rows := map[dk]*ReconRow{}
	for k, inv := range o.invoiced {
		r := rows[dk{k.Day, k.Exchange}]
		if r == nil {
			r = &ReconRow{Day: k.Day, Exchange: k.Exchange}
			rows[dk{k.Day, k.Exchange}] = r
		}
		ours := reconCell{}
		if c := o.billed[k]; c != nil {
			ours = *c
		}
		r.Invoice.Imps += inv.Imps
		r.Invoice.Spend += inv.Spend
		r.Ours.Imps += ours.Imps
		r.Ours.Spend += ours.Spend
		r.Placements = append(r.Placements, ReconLine{Placement: k.Placement, Ours: ours,
			Invoice: *inv, Gap: gap(ours.Spend, inv.Spend)})
	}
	// billed spend the exchange didn't invoice at all
	for k, c := range o.billed {
		if _, ok := o.invoiced[k]; ok {
			continue
		}
		if r := rows[dk{k.Day, k.Exchange}]; r != nil {
			r.Ours.Imps += c.Imps
			r.Ours.Spend += c.Spend
			r.Placements = append(r.Placements, ReconLine{Placement: k.Placement, Ours: *c, Gap: math.Inf(1)})
		}
	}
	out := make([]ReconRow, 0, len(rows))
	for _, r := range rows {
		r.Gap = gap(r.Ours.Spend, r.Invoice.Spend)
		sort.Slice(r.Placements, func(i, j int) bool { return r.Placements[i].Placement < r.Placements[j].Placement })
		for i := range r.Placements {
			if math.IsInf(r.Placements[i].Gap, 0) {
				r.Placements[i].Gap = 1
			}
		}
		out = append(out, *r)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Day != out[j].Day {
			return out[i].Day > out[j].Day
		}
		return out[i].Exchange < out[j].Exchange
	})
	return out
}

func gap(ours, invoice float64) float64 {
	if invoice == 0 {
		if ours == 0 {
			return 0
		}
		return 1
	}
	return (ours - invoice) / invoice
}

func (o *Ops) evaluateReconLocked(ts int64) {
	for _, r := range o.reconLocked() {
		worst := ""
		for _, p := range r.Placements {
			if math.Abs(p.Gap) > ReconTolerance && (worst == "" || math.Abs(p.Gap) > 0) {
				worst += fmt.Sprintf(" %s: ours $%.2f, invoiced $%.2f (%+.1f%%).", p.Placement,
					p.Ours.Spend, p.Invoice.Spend, 100*p.Gap)
			}
		}
		o.setAlertLocked("reconciliation_gap", fmt.Sprintf("%s, day %d", r.Exchange, r.Day),
			math.Abs(r.Gap) > ReconTolerance, ts, r.Gap, ReconTolerance,
			fmt.Sprintf("Billed spend $%.2f from our billing notices vs $%.2f invoiced (%+.1f%%).%s",
				r.Ours.Spend, r.Invoice.Spend, 100*r.Gap, worst))
	}
}
