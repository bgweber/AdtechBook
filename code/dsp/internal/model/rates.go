package model

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// InstallRates are smoothed click-to-install rates by segment (Chapter
// 10): what share of our clicks for a campaign led to an install, by
// country and placement type, each shrunk toward its parent's rate:
//
//	segment (campaign|country|placement) -> campaign -> all campaigns -> prior
//
// They're computed in batch from the MMP's install postbacks and published
// in the same bundle as the click model, so a deploy changes both.
type InstallRates struct {
	Prior     float64            `json:"prior"`
	K         float64            `json:"k"`
	Overall   float64            `json:"overall"`
	Campaigns map[string]float64 `json:"campaigns"`
	Segments  map[string]float64 `json:"segments"`
}

// LoadRates reads install_rates.json from a bundle; a bundle without one
// has no rates (nil, nil).
func LoadRates(dir string) (*InstallRates, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "install_rates.json"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	r := &InstallRates{}
	return r, json.Unmarshal(raw, r)
}

// SegmentKey is the key of a segment, in the FeatureVector's lowercase
// values, as the training code writes it.
func SegmentKey(campaign, country, placement string) string {
	return campaign + "|" + country + "|" + placement
}

// Lookup returns the most specific rate there is, and which level it
// came from: segment, campaign, overall, or prior.
func (r *InstallRates) Lookup(campaign, country, placement string) (float64, string) {
	if v, ok := r.Segments[SegmentKey(campaign, country, placement)]; ok {
		return v, "segment"
	}
	if v, ok := r.Campaigns[campaign]; ok {
		return v, "campaign"
	}
	if r.Overall > 0 {
		return r.Overall, "overall"
	}
	return r.Prior, "prior"
}

// ClickRates are the fallback when the click model can't be used
// (Chapter 12): smoothed click rates by placement type, country, and
// exchange, from the same impressions the model trained on, published in
// the same bundle:
//
//	placement|country|exchange -> placement|country -> placement -> overall
type ClickRates struct {
	K                float64            `json:"k"`
	Overall          float64            `json:"overall"`
	Placement        map[string]float64 `json:"placement"`
	PlacementCountry map[string]float64 `json:"placement_country"`
	Segments         map[string]float64 `json:"segments"`
}

// LoadClickRates reads click_rates.json from a bundle; nil if absent.
func LoadClickRates(dir string) (*ClickRates, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "click_rates.json"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	r := &ClickRates{}
	return r, json.Unmarshal(raw, r)
}

// Lookup returns the most specific click rate there is.
func (r *ClickRates) Lookup(placement, country, exchange string) float64 {
	if v, ok := r.Segments[placement+"|"+country+"|"+exchange]; ok {
		return v
	}
	if v, ok := r.PlacementCountry[placement+"|"+country]; ok {
		return v
	}
	if v, ok := r.Placement[placement]; ok {
		return v
	}
	return r.Overall
}
