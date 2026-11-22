package normalize

import (
	"encoding/json"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/openrtb"
)

func parse(t *testing.T, s string) *openrtb.BidRequest {
	t.Helper()
	var r openrtb.BidRequest
	if err := json.Unmarshal([]byte(s), &r); err != nil {
		t.Fatal(err)
	}
	return &r
}

// The two simulated exchanges describe the same rewarded video request
// differently; both normalize to the same internal request.
func TestExchangeQuirks(t *testing.T) {
	alpha := parse(t, `{"id":"r1","tmax":250,"imp":[{"id":"1","instl":1,"rwdd":1,
		"video":{"w":1080,"h":1920},"bidfloor":2.5}],
		"app":{"bundle":"com.gems.garden"},
		"device":{"ifa":"a-b-c","os":"android","geo":{"country":"USA"}}}`)
	beta := parse(t, `{"id":"r2","tmax":250,"imp":[{"id":"1","instl":1,
		"video":{"w":1080,"h":1920},"ext":{"is_rewarded_inventory":1},"bidfloor":2.5}],
		"app":{"bundle":"Com.Gems.Garden$hl=en"},
		"device":{"ifa":"a-b-c","os":"android","geo":{"country":"US"}}}`)
	a, err := Normalize("alpha_exchange", alpha)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Normalize("beta_exchange", beta)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []Request{a, b} {
		if n.Country != "USA" || n.Bundle != "com.gems.garden" ||
			n.Placement != RewardedVideo || !n.Rewarded || !n.ValidIFA {
			t.Errorf("%s: unexpected %+v", n.Exchange, n)
		}
	}
}

func TestPlacementsAndIDs(t *testing.T) {
	cases := []struct {
		json, placement string
		validIFA        bool
	}{
		{`{"id":"x","imp":[{"id":"1","banner":{"w":320,"h":50}}],"device":{"ifa":"00000000-0000-0000-0000-000000000000"}}`, Banner, false},
		{`{"id":"x","imp":[{"id":"1","instl":1,"video":{},"banner":{"w":320,"h":480}}],"device":{}}`, Interstitial, false},
		{`{"id":"x","imp":[{"id":"1","native":{"request":"{}"}}],"device":{"ifa":"d1"}}`, Native, true},
	}
	for _, c := range cases {
		n, err := Normalize("alpha_exchange", parse(t, c.json))
		if err != nil {
			t.Fatal(err)
		}
		if n.Placement != c.placement || n.ValidIFA != c.validIFA {
			t.Errorf("%s: got placement %q valid_ifa %v", c.json, n.Placement, n.ValidIFA)
		}
	}
	if _, err := Normalize("alpha_exchange", parse(t, `{"id":"x","imp":[]}`)); err != ErrNoImp {
		t.Errorf("want ErrNoImp, got %v", err)
	}
}
