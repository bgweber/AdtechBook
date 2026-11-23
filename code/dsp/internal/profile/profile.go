// Package profile keeps what the bidder has learned about each device
// from its own traffic: when it first and last saw the device, which apps
// it was seen in, and how many of our ads it has been shown.
//
// In crawl, this is the whole device profile. It lives in memory and is
// rebuilt from traffic after a restart. From walk on, it also counts the
// MMP's postbacks and is saved to disk across restarts (persist.go). A
// production store lives out of process (Chapter 17).
package profile

import (
	"hash/fnv"
	"sort"
	"sync"
)

const (
	shards    = 64
	maxApps   = 50 // distinct apps kept per device
	maxRecent = 20 // recent app visits kept per device
)

// Visit is one request from a device, in one app.
type Visit struct {
	App  string `json:"app"`
	TsMS int64  `json:"ts_ms"`
}

// Profile is one device's history, as the bidder saw it.
type Profile struct {
	FirstSeenMS int64
	LastSeenMS  int64
	Requests    int64
	LastImpMS   int64 // 0 if we never showed this device an ad
	ImpDay      int64
	ImpsToday   int64
	Installs    int64            // installs credited to us, from the MMP's postbacks
	Purchases   int64            // purchases after those installs
	apps        map[string]int64 // app -> last seen
	recent      []Visit          // most recent first
}

// Snapshot is a read-only copy for building features.
type Snapshot struct {
	Known       bool     `json:"known"`
	FirstSeenMS int64    `json:"first_seen_ms"`
	LastSeenMS  int64    `json:"last_seen_ms"`
	Requests    int64    `json:"requests"`
	LastImpMS   int64    `json:"last_imp_ms"`
	ImpsToday   int64    `json:"imps_today"`
	Installs    int64    `json:"installs,omitempty"` // postbacks received before this request
	Purchases   int64    `json:"purchases,omitempty"`
	Apps        []string `json:"apps"`   // sorted, deduplicated
	Recent      []Visit  `json:"recent"` // most recent first
}

type shard struct {
	mu sync.Mutex
	m  map[string]*Profile
}

// Store is an in-memory, sharded profile store.
type Store struct{ s [shards]shard }

func NewStore() *Store {
	st := &Store{}
	for i := range st.s {
		st.s[i].m = map[string]*Profile{}
	}
	return st
}

func (st *Store) shard(ifa string) *shard {
	h := fnv.New32a()
	h.Write([]byte(ifa))
	return &st.s[h.Sum32()%shards]
}

// Observe returns the device's profile as it was before this request,
// then records the request. Features use the snapshot, so they never
// include the request they describe.
func (st *Store) Observe(ifa, app string, tsMS, day int64) Snapshot {
	sh := st.shard(ifa)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	p, ok := sh.m[ifa]
	var snap Snapshot
	if ok {
		snap = p.snapshot(day)
	} else {
		p = &Profile{FirstSeenMS: tsMS, apps: map[string]int64{}}
		sh.m[ifa] = p
	}
	p.Requests++
	if tsMS > p.LastSeenMS {
		p.LastSeenMS = tsMS
	}
	if app != "" {
		if _, seen := p.apps[app]; !seen && len(p.apps) >= maxApps {
			p.evictOldestApp()
		}
		p.apps[app] = tsMS
		p.recent = append([]Visit{{app, tsMS}}, p.recent...)
		if len(p.recent) > maxRecent {
			p.recent = p.recent[:maxRecent]
		}
	}
	return snap
}

// Peek returns the device's profile without recording a request: for
// the load test, which replays requests without changing anything.
func (st *Store) Peek(ifa string, day int64) Snapshot {
	sh := st.shard(ifa)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if p, ok := sh.m[ifa]; ok {
		return p.snapshot(day)
	}
	return Snapshot{}
}

// RecordImpression notes that we showed the device an ad.
func (st *Store) RecordImpression(ifa string, tsMS, day int64) {
	sh := st.shard(ifa)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	p, ok := sh.m[ifa]
	if !ok {
		return
	}
	if p.ImpDay != day {
		p.ImpDay, p.ImpsToday = day, 0
	}
	p.ImpsToday++
	if tsMS > p.LastImpMS {
		p.LastImpMS = tsMS
	}
}

// RecordConversion notes an install or a purchase credited to us, when the
// MMP's postback arrives (Chapter 14). A feature built from it is as of the
// postback's arrival, the only time the bidder could know it.
func (st *Store) RecordConversion(ifa, event string) {
	sh := st.shard(ifa)
	sh.mu.Lock()
	defer sh.mu.Unlock()
	p, ok := sh.m[ifa]
	if !ok {
		return
	}
	switch event {
	case "install":
		p.Installs++
	case "af_purchase":
		p.Purchases++
	}
}

// SetConversions replaces every known device's install and purchase
// counts with recounted ones, such as after a batch of bad postbacks is
// quarantined (Chapter 24): a feature built from bad data has to be
// rebuilt too. A device missing from counts has none. It returns how many
// devices changed.
func (st *Store) SetConversions(counts map[string][2]int64) int {
	changed := 0
	for i := range st.s {
		sh := &st.s[i]
		sh.mu.Lock()
		for ifa, p := range sh.m {
			c := counts[ifa]
			if p.Installs != c[0] || p.Purchases != c[1] {
				changed++
				p.Installs, p.Purchases = c[0], c[1]
			}
		}
		sh.mu.Unlock()
	}
	return changed
}

// Devices returns how many devices the store knows.
func (st *Store) Devices() int {
	n := 0
	for i := range st.s {
		st.s[i].mu.Lock()
		n += len(st.s[i].m)
		st.s[i].mu.Unlock()
	}
	return n
}

func (p *Profile) snapshot(day int64) Snapshot {
	s := Snapshot{Known: true, FirstSeenMS: p.FirstSeenMS, LastSeenMS: p.LastSeenMS,
		Requests: p.Requests, LastImpMS: p.LastImpMS,
		Installs: p.Installs, Purchases: p.Purchases,
		Recent: append([]Visit(nil), p.recent...)}
	if p.ImpDay == day {
		s.ImpsToday = p.ImpsToday
	}
	for a := range p.apps {
		s.Apps = append(s.Apps, a)
	}
	sort.Strings(s.Apps)
	return s
}

func (p *Profile) evictOldestApp() {
	var oldest string
	var ts int64 = 1 << 62
	for a, t := range p.apps {
		if t < ts {
			oldest, ts = a, t
		}
	}
	delete(p.apps, oldest)
}
