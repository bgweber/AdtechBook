// Package features builds the FeatureVector: one model-agnostic record of
// clean feature values, logged with every bid (Chapter 8).
//
// The canonicalization rules live here and only here: natural units,
// normalized strings, explicit missing values (an absent key, never a
// silent zero), clipped impossible values, time relative to the request,
// and sets that are deduplicated and lists ordered most recent first.
// Every model turns this record into its own inputs.
package features

import (
	"strconv"
	"strings"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
	"github.com/bgweber/adtechbook/code/dsp/internal/normalize"
	"github.com/bgweber/adtechbook/code/dsp/internal/profile"
)

// SchemaVersion only grows: fields are added, never repurposed.
const SchemaVersion = 1

// StringList is a list with the age of each item, in seconds.
type StringList struct {
	Values     []string `json:"values"`
	AgeSeconds []int64  `json:"age_seconds"`
}

// FeatureVector follows the contract in Chapter 8.
type FeatureVector struct {
	SchemaVersion  int                   `json:"schema_version"`
	Numeric        map[string]float64    `json:"numeric"`
	Strings        map[string]string     `json:"strings"`
	Sets           map[string][]string   `json:"sets,omitempty"`
	Lists          map[string]StringList `json:"lists,omitempty"`
	LookupVersions map[string]string     `json:"lookup_versions,omitempty"`
}

// Build makes the FeatureVector for one request.
func Build(n *normalize.Request, p profile.Snapshot, tsMS int64) FeatureVector {
	fv := FeatureVector{
		SchemaVersion: SchemaVersion,
		Numeric:       map[string]float64{},
		Strings:       map[string]string{},
	}
	str := func(k, v string) {
		if v = strings.ToLower(strings.TrimSpace(v)); v != "" {
			fv.Strings[k] = v // absent key = missing
		}
	}
	str("exchange", n.Exchange)
	str("publisher_app", n.Bundle)
	str("placement", n.Placement)
	str("country", n.Country)
	str("os", n.OS)
	str("os_major", major(n.OSV))
	str("make", n.Make)
	str("model", n.Model)
	if n.ConnType > 0 {
		fv.Strings["connection_type"] = connType(n.ConnType)
	}

	fv.Numeric["bidfloor"] = max(n.Floor, 0)
	fv.Numeric["rewarded"] = b2f(n.Rewarded)
	fv.Numeric["interstitial"] = b2f(n.Instl)
	fv.Numeric["hour_of_day"] = float64((tsMS % clock.DayMS) / 3_600_000)

	// Device history, relative to this request. A device we've never
	// seen has no history features at all, rather than zeros.
	fv.Numeric["device_known"] = b2f(p.Known)
	if p.Known {
		fv.Numeric["device_requests"] = float64(p.Requests)
		fv.Numeric["seconds_since_first_seen"] = secs(tsMS - p.FirstSeenMS)
		fv.Numeric["seconds_since_last_seen"] = secs(tsMS - p.LastSeenMS)
		fv.Numeric["our_imps_today"] = float64(p.ImpsToday)
		if p.LastImpMS > 0 {
			fv.Numeric["seconds_since_our_last_imp"] = secs(tsMS - p.LastImpMS)
		}
		if len(p.Apps) > 0 {
			fv.Sets = map[string][]string{"apps_seen": p.Apps}
		}
		if len(p.Recent) > 0 {
			l := StringList{}
			for _, v := range p.Recent {
				l.Values = append(l.Values, v.App)
				l.AgeSeconds = append(l.AgeSeconds, int64(secs(tsMS-v.TsMS)))
			}
			fv.Lists = map[string]StringList{"recent_apps": l}
		}
	}
	return fv
}

func b2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// secs converts a duration to seconds, clipping negatives to zero: a
// late-arriving request can't be older than its own history.
func secs(ms int64) float64 {
	if ms < 0 {
		return 0
	}
	return float64(ms / 1000)
}

// SkewFineConnection injects a training-serving skew for the Chapter 8
// exercise: the bidder logs finer cellular connection types, such as
// "cellular_4g", while the training code still expects "cellular".
var SkewFineConnection bool

func major(v string) string {
	if i := strings.IndexByte(v, '.'); i >= 0 {
		return v[:i]
	}
	return v
}

func connType(c int) string {
	if SkewFineConnection {
		if g, ok := map[int]string{4: "cellular_2g", 5: "cellular_3g", 6: "cellular_4g", 7: "cellular_5g"}[c]; ok {
			return g
		}
	}
	switch c {
	case 2:
		return "wifi"
	case 3, 4, 5, 6, 7:
		return "cellular"
	case 1:
		return "ethernet"
	default:
		return "other"
	}
}

// Ad describes the campaign and creative being considered, for models
// that score them (Chapter 9). It's logged next to the FeatureVector, so
// training sees exactly the ad the bidder scored.
func Ad(c *campaigns.Campaign, cr campaigns.Creative) map[string]string {
	low := func(s string) string { return strings.ToLower(strings.TrimSpace(s)) }
	return map[string]string{
		"campaign":             strconv.Itoa(c.ID),
		"adv_app":              strconv.Itoa(c.AppID),
		"goal":                 low(c.Goal),
		"creative":             strconv.Itoa(cr.ID),
		"creative_format":      low(cr.Format),
		"creative_length":      strconv.Itoa(cr.LengthS),
		"creative_orientation": low(cr.Orientation),
		"creative_style":       low(cr.Style),
	}
}
