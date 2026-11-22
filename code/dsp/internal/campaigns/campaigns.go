// Package campaigns holds the campaign book: what the DSP is buying for
// whom, and how much it has spent today.
//
// Campaigns come from the campaign management system through the admin
// API (in the simulator, adsim pushes them when a run starts). Spend comes
// from billing notices, so the bidder's view of each budget is its own.
package campaigns

import (
	"encoding/json"
	"sort"
	"strings"
	"sync"
)

// Campaign is one advertiser campaign.
type Campaign struct {
	ID          int     `json:"campaign_id"`
	AppID       int     `json:"app_id"`       // the advertised app
	Goal        string  `json:"goal"`         // CPC, CPI, CPA, ROAS
	Target      float64 `json:"target"`       // dollars per goal event, or ROAS
	DailyBudget float64 `json:"daily_budget"` // dollars per day
	Geos        string  `json:"geos"`         // comma-separated country codes
	StartDay    int     `json:"start_day"`
	geoSet      map[string]bool
}

// Creative is one ad a campaign can show.
type Creative struct {
	ID          int    `json:"creative_id"`
	CampaignID  int    `json:"campaign_id"`
	Format      string `json:"format"` // video, playable, static
	LengthS     int    `json:"length_s"`
	Orientation string `json:"orientation"`
	Style       string `json:"style"`
}

// Compatible lists which creative formats can fill each placement.
var Compatible = map[string][]string{
	"rewarded_video": {"video", "playable"},
	"interstitial":   {"video", "playable", "static"},
	"banner":         {"static"},
	"native":         {"static", "video"},
}

// Book is the set of campaigns and creatives, plus spend by day.
type Book struct {
	mu        sync.RWMutex
	campaigns []*Campaign
	byID      map[int]*Campaign
	creatives map[[2]any][]Creative // (campaign ID, placement) -> creatives
	version   int
	spend     map[[2]int]float64 // (day, campaign ID) -> dollars
}

func NewBook() *Book {
	return &Book{byID: map[int]*Campaign{}, creatives: map[[2]any][]Creative{},
		spend: map[[2]int]float64{}}
}

// Load replaces the campaigns and creatives. Spend so far is kept.
func (b *Book) Load(camps []Campaign, crs []Creative) {
	byID := map[int]*Campaign{}
	list := make([]*Campaign, 0, len(camps))
	for i := range camps {
		c := camps[i]
		c.geoSet = map[string]bool{}
		for _, g := range strings.Split(c.Geos, ",") {
			c.geoSet[strings.TrimSpace(g)] = true
		}
		byID[c.ID] = &c
		list = append(list, &c)
	}
	sort.Slice(list, func(i, j int) bool { return list[i].ID < list[j].ID })
	fits := map[[2]any][]Creative{}
	for _, cr := range crs {
		for placement, formats := range Compatible {
			for _, f := range formats {
				if cr.Format == f {
					k := [2]any{cr.CampaignID, placement}
					fits[k] = append(fits[k], cr)
				}
			}
		}
	}
	b.mu.Lock()
	b.campaigns, b.byID, b.creatives = list, byID, fits
	b.version++
	b.mu.Unlock()
}

// Candidate is an eligible campaign with its creatives for the placement.
type Candidate struct {
	Campaign  *Campaign
	Creatives []Creative
	Remaining float64 // share of today's budget left, 0 to 1
}

// Eligible returns the campaigns that can bid on a request: targeting the
// country, started, with budget left, and with a creative that fits.
func (b *Book) Eligible(country, placement string, day int) []Candidate {
	out, _ := b.EligibleWhy(country, placement, day)
	return out
}

// Filter reasons, in the order the filters run (Chapter 6).
const (
	FilterGeo        = "geo"
	FilterNotStarted = "not_started"
	FilterNoCreative = "no_fitting_creative"
	FilterBudget     = "budget_spent"
)

// EligibleWhy is Eligible, plus the filter that removed each campaign
// that isn't eligible, so the bid ledger can say which campaigns were
// filtered and why.
func (b *Book) EligibleWhy(country, placement string, day int) ([]Candidate, map[int]string) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	var out []Candidate
	why := map[int]string{}
	for _, c := range b.campaigns {
		if !c.geoSet[country] {
			why[c.ID] = FilterGeo
			continue
		}
		if day < c.StartDay {
			why[c.ID] = FilterNotStarted
			continue
		}
		crs := b.creatives[[2]any{c.ID, placement}]
		if len(crs) == 0 {
			why[c.ID] = FilterNoCreative
			continue
		}
		left := 1 - b.spend[[2]int{day, c.ID}]/c.DailyBudget
		if left <= 0 {
			why[c.ID] = FilterBudget
			continue
		}
		out = append(out, Candidate{Campaign: c, Creatives: crs, Remaining: left})
	}
	return out, why
}

// AddSpend records a billed impression's cost.
func (b *Book) AddSpend(day, campaignID int, dollars float64) {
	b.mu.Lock()
	b.spend[[2]int{day, campaignID}] += dollars
	b.mu.Unlock()
}

// CampaignView is one campaign as the console shows it.
type CampaignView struct {
	Campaign
	SpendToday float64 `json:"spend_today"`
	Creatives  int     `json:"creatives"`
}

// Views returns every campaign with today's spend.
func (b *Book) Views(day int) []CampaignView {
	b.mu.RLock()
	defer b.mu.RUnlock()
	n := map[int]int{}
	for k, crs := range b.creatives {
		if k[1] == "interstitial" { // every format fits interstitials
			n[k[0].(int)] = len(crs)
		}
	}
	out := make([]CampaignView, 0, len(b.campaigns))
	for _, c := range b.campaigns {
		out = append(out, CampaignView{Campaign: *c, SpendToday: b.spend[[2]int{day, c.ID}],
			Creatives: n[c.ID]})
	}
	return out
}

func (b *Book) Version() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.version
}

// LoadJSON reads {"campaigns": [...], "creatives": [...]}.
func (b *Book) LoadJSON(data []byte) error {
	var in struct {
		Campaigns []Campaign `json:"campaigns"`
		Creatives []Creative `json:"creatives"`
	}
	if err := json.Unmarshal(data, &in); err != nil {
		return err
	}
	b.Load(in.Campaigns, in.Creatives)
	return nil
}
