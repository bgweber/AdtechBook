package policy

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
)

// Calibration holds the calibration multipliers, kappa, by campaign
// (Chapter 21): realized installs over predicted installs on recent
// matured wins, shrunk toward the overall value and bounded. The training
// service computes them; the bidder multiplies each campaign's advertiser
// value by its kappa.
type Calibration struct {
	Version    string             `json:"version"`
	Overall    float64            `json:"overall"`
	ByCampaign map[string]float64 `json:"by_campaign"`
}

// Kappas is the bidder's current calibration, swapped atomically.
type Kappas struct{ cur atomic.Pointer[Calibration] }

// Get is a campaign's multiplier: 1 when there's no calibration.
func (k *Kappas) Get(campaignID int) float64 {
	if k == nil {
		return 1
	}
	c := k.cur.Load()
	if c == nil {
		return 1
	}
	if v, ok := c.ByCampaign[itoa(campaignID)]; ok && v > 0 {
		return v
	}
	if c.Overall > 0 {
		return c.Overall
	}
	return 1
}

// Current returns the loaded calibration, or nil.
func (k *Kappas) Current() *Calibration { return k.cur.Load() }

// Load reads the newest calibration file under dir, such as
// out/models/calibration/kappa-0003.json. No file is no calibration.
func (k *Kappas) Load(dir string) (*Calibration, error) {
	files, _ := filepath.Glob(filepath.Join(dir, "kappa-*.json"))
	if len(files) == 0 {
		return nil, nil
	}
	sort.Strings(files)
	raw, err := os.ReadFile(files[len(files)-1])
	if err != nil {
		return nil, err
	}
	var c Calibration
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	for id, v := range c.ByCampaign {
		c.ByCampaign[id] = math.Min(2, math.Max(0.5, v)) // bounded, as in the chapter
	}
	k.cur.Store(&c)
	return &c, nil
}

// Off goes back to no calibration.
func (k *Kappas) Off() { k.cur.Store(nil) }

// Creatives keeps each creative's impressions and installs, by day of
// impression, for Thompson sampling (Chapter 21). An impression's installs
// count once it is a full day old, so recent impressions don't count as
// failures before their installs could arrive.
type Creatives struct {
	mu    sync.Mutex
	stats map[int]map[int64]*cell // creative -> impression day -> counts
	camp  map[int]int             // creative -> campaign
	imps  map[string]impRef       // request ID -> its creative, for postbacks
	on    atomic.Bool
	today atomic.Int64 // the newest impression day seen

	pmu   sync.Mutex
	picks map[string]*pickCache // by the campaign's creative IDs (Chapter 17)
}

// pickCache holds one campaign's posteriors and each creative's chance of
// being chosen, for PickRefresh picks or until the day changes. The
// posteriors only move as day-old installs arrive, slowly, so rebuilding
// them, and estimating the chances from fresh draws, on every request was
// a seventh of the bidder's CPU (Chapter 17). Draws and logged chances
// both come from the cached posteriors, so they agree.
type pickCache struct {
	today int64
	uses  int
	post  []Posterior
	prob  []float64
}

// PickRefresh is how many picks a campaign's cached posteriors serve.
const PickRefresh = 1000

// pickDraws is how many draws estimate the chances of each creative.
const pickDraws = 4000

type cell struct{ Imps, Installs float64 }

type impRef struct {
	Creative int
	Day      int64
}

// NewCreatives returns empty creative stats, with Thompson sampling off.
func NewCreatives() *Creatives {
	return &Creatives{stats: map[int]map[int64]*cell{}, camp: map[int]int{}, imps: map[string]impRef{}}
}

// Reset forgets every creative's counts, to start a creative test fresh.
func (c *Creatives) Reset() {
	c.mu.Lock()
	c.stats, c.camp, c.imps = map[int]map[int64]*cell{}, map[int]int{}, map[string]impRef{}
	c.mu.Unlock()
	c.pmu.Lock()
	c.picks = nil
	c.pmu.Unlock()
}

// SetThompson turns Thompson sampling over creatives on or off; off is
// uniform rotation, as in crawl.
func (c *Creatives) SetThompson(on bool) { c.on.Store(on) }

// Thompson reports whether Thompson sampling is on.
func (c *Creatives) Thompson() bool { return c != nil && c.on.Load() }

// Impression records a billed impression.
func (c *Creatives) Impression(requestID string, creative, campaign int, day int64) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cellLocked(creative, day).Imps++
	if day > c.today.Load() {
		c.today.Store(day)
	}
	c.camp[creative] = campaign
	c.imps[requestID] = impRef{creative, day}
	if len(c.imps) > 2_000_000 {
		c.pruneLocked(day - 8)
	}
}

// Install records an install attributed to one of our impressions.
func (c *Creatives) Install(requestID string) {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if ref, ok := c.imps[requestID]; ok {
		c.cellLocked(ref.Creative, ref.Day).Installs++
	}
}

func (c *Creatives) cellLocked(creative int, day int64) *cell {
	m := c.stats[creative]
	if m == nil {
		m = map[int64]*cell{}
		c.stats[creative] = m
	}
	x := m[day]
	if x == nil {
		x = &cell{}
		m[day] = x
	}
	return x
}

func (c *Creatives) pruneLocked(before int64) {
	for id, r := range c.imps {
		if r.Day < before {
			delete(c.imps, id)
		}
	}
}

// Posterior is one creative's Beta posterior over installs per impression.
type Posterior struct {
	Creative int     `json:"creative_id"`
	Imps     float64 `json:"impressions"`
	Installs float64 `json:"installs"`
	A        float64 `json:"alpha"`
	B        float64 `json:"beta"`
	Share    float64 `json:"share,omitempty"` // estimated chance it's chosen
}

// PriorStrength is how many impressions of evidence the prior is worth.
const PriorStrength = 1000

// posteriors builds the Beta posteriors for a campaign's creatives, from
// impressions at least a day old. The prior comes from the campaign's
// creatives pooled, worth PriorStrength impressions.
func (c *Creatives) posteriors(ids []int, today int64) []Posterior {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Posterior, len(ids))
	var ti, tk float64
	for i, id := range ids {
		out[i].Creative = id
		for day, x := range c.stats[id] {
			if day < today { // a full day old: its installs have mostly arrived
				out[i].Imps += x.Imps
				out[i].Installs += x.Installs
			}
		}
		ti += out[i].Imps
		tk += out[i].Installs
	}
	rate := 0.002
	if ti > 0 && tk > 0 {
		rate = tk / ti
	}
	for i := range out {
		out[i].A = 1 + rate*PriorStrength + out[i].Installs
		out[i].B = 1 + (1-rate)*PriorStrength + math.Max(0, out[i].Imps-out[i].Installs)
	}
	return out
}

// Pick chooses a creative: one draw from each posterior, and the largest
// wins. It returns the index, the estimated probability of choosing it
// (to log for offline evaluation), and "thompson"; or a uniform choice and
// "random" when Thompson sampling is off.
func (c *Creatives) Pick(ids []int, rng *rand.Rand) (int, float64, string) {
	n := len(ids)
	if !c.Thompson() || n < 2 {
		return rng.IntN(n), 1 / float64(n), "random"
	}
	pc := c.cached(ids, rng)
	i := draw(pc.post, rng)
	return i, pc.prob[i], "thompson"
}

// draw takes one draw from each posterior and returns the largest's index.
func draw(post []Posterior, rng *rand.Rand) int {
	best, bi := -1.0, 0
	for i, p := range post {
		if v := betaSample(rng, p.A, p.B); v > best {
			best, bi = v, i
		}
	}
	return bi
}

// cached returns a campaign's posteriors and chances, rebuilt when stale.
func (c *Creatives) cached(ids []int, rng *rand.Rand) *pickCache {
	today := c.today.Load()
	key := fmt.Sprint(ids)
	c.pmu.Lock()
	defer c.pmu.Unlock()
	pc := c.picks[key]
	if pc == nil || pc.today != today || pc.uses >= PickRefresh {
		pc = &pickCache{today: today, post: c.posteriors(ids, today), prob: make([]float64, len(ids))}
		for k := 0; k < pickDraws; k++ {
			pc.prob[draw(pc.post, rng)]++
		}
		for i := range pc.prob { // smoothed, so no logged chance is zero
			pc.prob[i] = (pc.prob[i] + 1) / float64(pickDraws+len(ids))
		}
		if c.picks == nil {
			c.picks = map[string]*pickCache{}
		}
		c.picks[key] = pc
	}
	pc.uses++
	return pc
}

// Report returns every campaign's creative posteriors, with each
// creative's estimated share of draws, for the console.
func (c *Creatives) Report() map[int][]Posterior {
	today := c.today.Load()
	c.mu.Lock()
	byCamp := map[int][]int{}
	for id, camp := range c.camp {
		byCamp[camp] = append(byCamp[camp], id)
	}
	c.mu.Unlock()
	out := map[int][]Posterior{}
	rng := rand.New(rand.NewPCG(1, 2))
	for camp, ids := range byCamp {
		sort.Ints(ids)
		post := c.posteriors(ids, today)
		wins := make([]int, len(ids))
		for k := 0; k < 2000; k++ {
			best, bi := -1.0, 0
			for i, p := range post {
				if v := betaSample(rng, p.A, p.B); v > best {
					best, bi = v, i
				}
			}
			wins[bi]++
		}
		for i := range post {
			post[i].Share = float64(wins[i]) / 2000
		}
		out[camp] = post
	}
	return out
}

// saved is the state Save writes: counts, not the request map.
type savedCreatives struct {
	Stats map[int]map[int64]*cell `json:"stats"`
	Camp  map[int]int             `json:"campaigns"`
	On    bool                    `json:"thompson"`
}

// Save writes the counts, so a restart doesn't forget what each creative
// has shown.
func (c *Creatives) Save(path string) error {
	c.mu.Lock()
	raw, err := json.Marshal(savedCreatives{c.stats, c.camp, c.on.Load()})
	c.mu.Unlock()
	if err != nil {
		return err
	}
	os.MkdirAll(filepath.Dir(path), 0o755)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Load reads counts written by Save; a missing file is no data.
func (c *Creatives) Load(path string) error {
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}
	var s savedCreatives
	if err := json.Unmarshal(raw, &s); err != nil {
		return err
	}
	c.pmu.Lock()
	c.picks = nil
	c.pmu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()
	if s.Stats != nil {
		c.stats = s.Stats
	}
	if s.Camp != nil {
		c.camp = s.Camp
	}
	for _, days := range c.stats {
		for d := range days {
			if d > c.today.Load() {
				c.today.Store(d)
			}
		}
	}
	c.on.Store(s.On)
	return nil
}

// betaSample draws from Beta(a, b) as X / (X + Y), with X and Y Gamma.
func betaSample(rng *rand.Rand, a, b float64) float64 {
	x, y := gammaSample(rng, a), gammaSample(rng, b)
	return x / (x + y)
}

// gammaSample draws from Gamma(a, 1): Marsaglia and Tsang's method.
func gammaSample(rng *rand.Rand, a float64) float64 {
	if a < 1 {
		return gammaSample(rng, a+1) * math.Pow(rng.Float64(), 1/a)
	}
	d := a - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x || math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}

func itoa(i int) string { return strconv.Itoa(i) }
