package features

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/normalize"
	"github.com/bgweber/adtechbook/code/dsp/internal/openrtb"
	"github.com/bgweber/adtechbook/code/dsp/internal/profile"
)

// testdata/parity_cases.jsonl holds raw contexts from a simulator run,
// with the FeatureVector the bidder logged for each. The Python parity
// check (code/training/adtrain/features_ref.py) has a test on the same
// file, so the two implementations can't drift apart without a test
// failing on one side.
func TestParityCases(t *testing.T) {
	f, err := os.Open(filepath.Join("testdata", "parity_cases.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<22)
	n := 0
	for sc.Scan() {
		var c struct {
			Exchange string             `json:"exchange"`
			TsMS     int64              `json:"ts_ms"`
			Request  openrtb.BidRequest `json:"request"`
			Profile  profile.Snapshot   `json:"profile"`
			Features FeatureVector      `json:"features"`
		}
		if err := json.Unmarshal(sc.Bytes(), &c); err != nil {
			t.Fatal(err)
		}
		req, err := normalize.Normalize(c.Exchange, &c.Request)
		if err != nil {
			t.Fatal(err)
		}
		got := Build(&req, c.Profile, c.TsMS)
		if !reflect.DeepEqual(got, c.Features) {
			t.Errorf("case %d (%s): features differ\n got %+v\nwant %+v", n, c.Request.ID, got, c.Features)
		}
		n++
	}
	if n < 10 {
		t.Fatalf("only %d cases", n)
	}
}
