// Package ops is the crawl DSP's operations layer (Chapter 12): kill
// switches with an audit log, even-spend throttling, a few alerts that
// matter, and spend reconciliation with exchanges.
//
// Everything here runs on simulated time, from the X-Adsim-Time header,
// so a simulated day of operations takes a minute of wall time.
package ops

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

// Switches are the kill switches of Chapter 12, at each level. A switch
// that's on turns the thing off: stop all bidding, stop one exchange or
// campaign, take the model out of pricing (the bidder falls back to
// segment click rates), or treat a feature as missing everywhere.
type Switches struct {
	Global    bool            `json:"global"`
	Model     bool            `json:"model"`
	Exchanges map[string]bool `json:"exchanges"`
	Campaigns map[int]bool    `json:"campaigns"`
	Features  map[string]bool `json:"features"`
	Version   int             `json:"version"`
}

// AuditEntry is one change to the switches or the configuration.
type AuditEntry struct {
	At      string `json:"at"`      // wall clock, UTC
	SimMS   int64  `json:"sim_ms"`  // simulated time of the change
	What    string `json:"what"`    // such as "campaign 3 paused"
	Version int    `json:"version"` // switches version after the change
	Who     string `json:"who"`     // console, api, startup
}

type switchboard struct {
	dir   string
	cur   atomic.Pointer[Switches]
	mu    sync.Mutex
	audit []AuditEntry
}

func (s *switchboard) load() {
	sw := &Switches{Exchanges: map[string]bool{}, Campaigns: map[int]bool{}, Features: map[string]bool{}}
	if raw, err := os.ReadFile(filepath.Join(s.dir, "switches.json")); err == nil {
		json.Unmarshal(raw, sw)
		if sw.Exchanges == nil {
			sw.Exchanges = map[string]bool{}
		}
		if sw.Campaigns == nil {
			sw.Campaigns = map[int]bool{}
		}
		if sw.Features == nil {
			sw.Features = map[string]bool{}
		}
	}
	s.cur.Store(sw)
	if raw, err := os.ReadFile(filepath.Join(s.dir, "audit.jsonl")); err == nil {
		for _, line := range splitLines(raw) {
			var e AuditEntry
			if json.Unmarshal(line, &e) == nil {
				s.audit = append(s.audit, e)
			}
		}
	}
}

// Switches returns the current switches. It's one atomic load, so the
// bid path never waits on a change.
func (o *Ops) Switches() *Switches { return o.sw.cur.Load() }

// Set flips one switch. kind is global, model, exchange, campaign, or
// feature; key names the exchange, campaign, or feature. It takes effect
// on the next request, and is written to the audit log.
func (o *Ops) Set(kind, key string, on bool, who string, simMS int64) (*Switches, error) {
	s := &o.sw
	s.mu.Lock()
	defer s.mu.Unlock()
	old := s.cur.Load()
	n := &Switches{Global: old.Global, Model: old.Model, Version: old.Version + 1,
		Exchanges: copyMap(old.Exchanges), Campaigns: copyMap(old.Campaigns), Features: copyMap(old.Features)}
	verb := map[bool]string{true: "off", false: "back on"}[on]
	var what string
	switch kind {
	case "global":
		n.Global = on
		what = "all bidding " + map[bool]string{true: "stopped", false: "resumed"}[on]
	case "model":
		n.Model = on
		what = "click model " + map[bool]string{true: "switched off: pricing falls back to segment click rates",
			false: "switched back on"}[on]
	case "exchange":
		if key == "" {
			return old, fmt.Errorf("no exchange given")
		}
		setOrDelete(n.Exchanges, key, on)
		what = "exchange " + key + " " + verb
	case "campaign":
		var id int
		if _, err := fmt.Sscan(key, &id); err != nil {
			return old, fmt.Errorf("bad campaign %q", key)
		}
		setOrDelete(n.Campaigns, id, on)
		what = "campaign " + key + " " + map[bool]string{true: "paused", false: "resumed"}[on]
	case "feature":
		if key == "" {
			return old, fmt.Errorf("no feature given")
		}
		setOrDelete(n.Features, key, on)
		what = "feature " + key + " " + map[bool]string{true: "treated as missing", false: "back in use"}[on]
	default:
		return old, fmt.Errorf("unknown switch %q", kind)
	}
	s.cur.Store(n)
	o.auditLocked(what, n.Version, who, simMS)
	o.persistLocked(n)
	return n, nil
}

// Note records a configuration change that isn't a switch, such as a new
// campaign book, so "what changed?" has an answer (Chapter 12).
func (o *Ops) Note(what, who string, simMS int64) {
	if simMS == 0 { // the console's changes happen at the newest simulated time seen
		o.mu.Lock()
		simMS = o.nowMS
		o.mu.Unlock()
	}
	o.sw.mu.Lock()
	defer o.sw.mu.Unlock()
	o.auditLocked(what, o.sw.cur.Load().Version, who, simMS)
}

// Audit returns the audit log, newest first.
func (o *Ops) Audit() []AuditEntry {
	o.sw.mu.Lock()
	defer o.sw.mu.Unlock()
	out := make([]AuditEntry, len(o.sw.audit))
	for i, e := range o.sw.audit {
		out[len(out)-1-i] = e
	}
	return out
}

func (o *Ops) auditLocked(what string, version int, who string, simMS int64) {
	e := AuditEntry{At: time.Now().UTC().Format(time.RFC3339), SimMS: simMS, What: what,
		Version: version, Who: who}
	o.sw.audit = append(o.sw.audit, e)
	if o.sw.dir == "" {
		return
	}
	os.MkdirAll(o.sw.dir, 0o755)
	f, err := os.OpenFile(filepath.Join(o.sw.dir, "audit.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err == nil {
		json.NewEncoder(f).Encode(e)
		f.Close()
	}
}

func (o *Ops) persistLocked(sw *Switches) {
	if o.sw.dir == "" {
		return
	}
	os.MkdirAll(o.sw.dir, 0o755)
	raw, _ := json.MarshalIndent(sw, "", " ")
	tmp := filepath.Join(o.sw.dir, "switches.json.tmp")
	if os.WriteFile(tmp, raw, 0o644) == nil {
		os.Rename(tmp, filepath.Join(o.sw.dir, "switches.json"))
	}
}

func copyMap[K comparable](m map[K]bool) map[K]bool {
	out := make(map[K]bool, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func setOrDelete[K comparable](m map[K]bool, k K, on bool) {
	if on {
		m[k] = true
	} else {
		delete(m, k)
	}
}

func splitLines(b []byte) [][]byte {
	var out [][]byte
	start := 0
	for i, c := range b {
		if c == '\n' {
			if i > start {
				out = append(out, b[start:i])
			}
			start = i + 1
		}
	}
	if start < len(b) {
		out = append(out, b[start:])
	}
	return out
}
