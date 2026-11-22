// Package normalize turns an exchange's bid request into the one internal
// form the rest of the bidder uses.
//
// Every exchange has quirks. In the simulated market, one sends
// three-letter country codes and another sends two-letter ones, one flags
// rewarded video with the standard imp.rwdd field and another in imp.ext,
// and some append junk to app bundles. Normalizing once, at the edge, means
// features, eligibility, and logs never see those differences
// (Chapter 6).
package normalize

import (
	"encoding/json"
	"errors"
	"strings"

	"github.com/bgweber/adtechbook/code/dsp/internal/openrtb"
)

// Placement types the bidder knows how to fill.
const (
	RewardedVideo = "rewarded_video"
	Interstitial  = "interstitial"
	Banner        = "banner"
	Native        = "native"
)

// ZeroIFA is the device ID a device sends when ad tracking is limited.
const ZeroIFA = "00000000-0000-0000-0000-000000000000"

// alpha2To3 maps two-letter country codes to the three-letter codes the
// bidder uses everywhere. It covers the simulator's markets; a production
// bidder loads the full ISO 3166 table.
var alpha2To3 = map[string]string{
	"IN": "IND", "BR": "BRA", "ID": "IDN", "MX": "MEX",
	"US": "USA", "JP": "JPN", "DE": "DEU",
}

// Request is a normalized bid request.
type Request struct {
	ID        string  `json:"request_id"`
	Exchange  string  `json:"exchange"`
	TMax      int     `json:"tmax"`
	Bundle    string  `json:"bundle"`
	Placement string  `json:"placement"`
	Rewarded  bool    `json:"rewarded"`
	Instl     bool    `json:"instl"`
	Floor     float64 `json:"bidfloor"`
	Country   string  `json:"country"`
	OS        string  `json:"os"`
	OSV       string  `json:"osv"`
	Make      string  `json:"make"`
	Model     string  `json:"model"` // "" when the exchange didn't send it
	ConnType  int     `json:"connectiontype"`
	IFA       string  `json:"ifa"`
	ValidIFA  bool    `json:"valid_ifa"`
}

// ErrNoImp means the request has nothing to bid on.
var ErrNoImp = errors.New("request has no impressions")

// Normalize maps one exchange's request to the internal form.
func Normalize(exchange string, r *openrtb.BidRequest) (Request, error) {
	if len(r.Imp) == 0 {
		return Request{}, ErrNoImp
	}
	imp := r.Imp[0]
	n := Request{
		ID:        r.ID,
		Exchange:  exchange,
		TMax:      r.TMax,
		Floor:     imp.BidFloor,
		Instl:     imp.Instl == 1,
		Rewarded:  imp.Rwdd == 1 || extRewarded(imp.Ext),
		Placement: placement(imp),
	}
	if r.App != nil {
		n.Bundle = cleanBundle(r.App.Bundle)
	}
	if d := r.Device; d != nil {
		n.OS, n.OSV, n.Make, n.Model = d.OS, d.OSV, d.Make, d.Model
		n.ConnType = d.ConnectionType
		n.IFA = d.IFA
		n.ValidIFA = d.IFA != "" && d.IFA != ZeroIFA
		if d.Geo != nil {
			n.Country = country3(d.Geo.Country)
		}
	}
	return n, nil
}

// placement infers the ad format from which objects the imp carries.
func placement(imp openrtb.Imp) string {
	switch {
	case imp.Native != nil:
		return Native
	case imp.Video != nil && (imp.Rwdd == 1 || extRewarded(imp.Ext)):
		return RewardedVideo
	case imp.Video != nil:
		return Interstitial
	default:
		return Banner
	}
}

// extRewarded reads the rewarded flag some exchanges put in imp.ext.
func extRewarded(ext json.RawMessage) bool {
	if len(ext) == 0 {
		return false
	}
	var e struct {
		IsRewarded int `json:"is_rewarded_inventory"`
	}
	return json.Unmarshal(ext, &e) == nil && e.IsRewarded == 1
}

// SkewKeepBundleSuffix injects a training-serving skew for the
// Chapter 8 exercise: the bidder stops dropping the junk some exchanges
// append to bundles, while the training code still drops it.
var SkewKeepBundleSuffix bool

// cleanBundle drops junk some exchanges append, such as "$hl=en", and
// lowercases the bundle so the same app always has the same key.
func cleanBundle(b string) string {
	if i := strings.IndexByte(b, '$'); i >= 0 && !SkewKeepBundleSuffix {
		b = b[:i]
	}
	return strings.ToLower(strings.TrimSpace(b))
}

func country3(c string) string {
	c = strings.ToUpper(strings.TrimSpace(c))
	if len(c) == 2 {
		if c3, ok := alpha2To3[c]; ok {
			return c3
		}
	}
	return c
}
