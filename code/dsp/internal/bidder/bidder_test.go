package bidder

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/clock"
	"github.com/bgweber/adtechbook/code/dsp/internal/features"
	"github.com/bgweber/adtechbook/code/dsp/internal/ledger"
	"github.com/bgweber/adtechbook/code/dsp/internal/model"
	"github.com/bgweber/adtechbook/code/dsp/internal/openrtb"
	"github.com/bgweber/adtechbook/code/dsp/internal/ops"
	"github.com/bgweber/adtechbook/code/dsp/internal/policy"
	"github.com/bgweber/adtechbook/code/dsp/internal/profile"
	"github.com/bgweber/adtechbook/code/dsp/internal/stats"
)

func newBidder(t *testing.T, mode string) *Bidder {
	t.Helper()
	dir := t.TempDir()
	bids, err := ledger.New(dir, "bids")
	if err != nil {
		t.Fatal(err)
	}
	notices, err := ledger.New(dir, "notices")
	if err != nil {
		t.Fatal(err)
	}
	posts, err := ledger.New(dir, "postbacks")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(posts.Close)
	clicks, err := ledger.New(dir, "clicks")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(bids.Close)
	t.Cleanup(notices.Close)
	t.Cleanup(clicks.Close)
	book := campaigns.NewBook()
	book.Load([]campaigns.Campaign{{ID: 7, Goal: "CPC", Target: 0.10, DailyBudget: 5, Geos: "BRA"}},
		[]campaigns.Creative{{ID: 70, CampaignID: 7, Format: "static"}})
	p := policy.DefaultCrawl()
	p.ExploreShare, p.HoldoutPerMil = 0, 0
	return &Bidder{Mode: mode, Version: "test", BidShare: 1, NoticeURL: "http://dsp",
		Policy: p, Book: book, Profiles: profile.NewStore(), Bids: bids, Notices: notices,
		Clicks: clicks, Postbacks: posts, MMPToken: "secret", Stats: stats.New(),
		Models: &model.Registry{Dir: t.TempDir()}, Ops: ops.New(t.TempDir())}
}

func do(b *Bidder, method, path, body, simMS string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	if simMS != "" {
		r.Header.Set(clock.Header, simMS)
	}
	w := httptest.NewRecorder()
	b.ServeHTTP(w, r)
	return w
}

const banner = `{"id":"r1","tmax":250,"imp":[{"id":"1","banner":{"w":320,"h":50}}],
	"app":{"bundle":"com.gems.garden"},"device":{"ifa":"d1","geo":{"country":"BR"}}}`

// In listen mode every request gets an empty 204, and the reasons say why.
func TestListenModeNeverBids(t *testing.T) {
	b := newBidder(t, ModeListen)
	reqs := []string{
		banner,
		`{"id":"b","imp":[{"id":"1","banner":{}}],"device":{"ifa":"00000000-0000-0000-0000-000000000000"}}`,
		`not json`,
	}
	for _, body := range reqs {
		if w := do(b, "POST", "/openrtb/beta_exchange", body, "90000000"); w.Code != http.StatusNoContent {
			t.Errorf("want 204, got %d %q", w.Code, w.Body.String())
		}
	}
	s := b.Stats.Snapshot()
	for reason, want := range map[string]int64{"listen": 1, "no_device_id": 1, "bad_request": 1} {
		if s.Reasons[reason] != want {
			t.Errorf("reason %s: got %d, want %d", reason, s.Reasons[reason], want)
		}
	}
	// 90,000,000 ms is day 1, 01:00 in simulated time.
	if s.SimDay != 1 || s.Hourly[1] != 3 {
		t.Errorf("simulated clock not used: day %d, hourly %v", s.SimDay, s.Hourly)
	}
}

// A bid, its win and billing notices, and the spend they add up to.
func TestBidAndNotices(t *testing.T) {
	b := newBidder(t, ModeBid)
	w := do(b, "POST", "/openrtb/beta_exchange", banner, "1000")
	if w.Code != http.StatusOK {
		t.Fatalf("want a bid, got %d", w.Code)
	}
	var resp openrtb.BidResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	bid := resp.SeatBid[0].Bid[0]
	// CPC $0.10 x 0.6% banner click rate x 1000 = $0.60 CPM value, minus
	// the 20% margin = $0.48, shaded by 0.7 = $0.336.
	if bid.CID != "7" || bid.CrID != "70" || bid.Price != 0.336 {
		t.Fatalf("unexpected bid %+v", bid)
	}
	fire := func(u string) {
		u = strings.NewReplacer("${AUCTION_PRICE}", "0.336", "${AUCTION_MIN_TO_WIN}", "",
			"${AUCTION_LOSS}", "").Replace(u)
		p, _ := url.Parse(u)
		if w := do(b, "GET", p.RequestURI(), "", "2000"); w.Code != http.StatusNoContent {
			t.Fatalf("notice %s: got %d", u, w.Code)
		}
	}
	fire(bid.NURL)
	fire(bid.BURL)
	fire(bid.BURL) // an exchange retry: logged, not double-counted
	s := b.Stats.Snapshot()
	c := s.Campaigns[7]
	if c.Bids != 1 || c.Wins != 1 || c.Imps != 1 || s.DupNotices != 1 {
		t.Errorf("unexpected counts: %+v, duplicates %d", c, s.DupNotices)
	}
	if v := b.Book.Views(0)[0]; math.Abs(v.SpendToday-0.000336) > 1e-12 {
		t.Errorf("spend today: got %v, want 0.000336", v.SpendToday)
	}
	if math.Abs(c.PredClicks-0.006) > 1e-12 {
		t.Errorf("predicted clicks: got %v, want the banner prior 0.006", c.PredClicks)
	}

	// The ad markup carries the click tracker, as an ad SDK would call it.
	var adm struct {
		Native struct {
			Link struct {
				ClickTrackers []string `json:"clicktrackers"`
			} `json:"link"`
		} `json:"native"`
	}
	if err := json.Unmarshal([]byte(bid.AdM), &adm); err != nil || len(adm.Native.Link.ClickTrackers) != 1 {
		t.Fatalf("no click tracker in adm %q: %v", bid.AdM, err)
	}
	fire(adm.Native.Link.ClickTrackers[0])
	fire(adm.Native.Link.ClickTrackers[0]) // a retry: a duplicate, not a second click
	s = b.Stats.Snapshot()
	if c := s.Campaigns[7]; c.Clicks != 1 || s.DupClicks != 1 {
		t.Errorf("clicks: got %d (duplicates %d), want 1 (1)", c.Clicks, s.DupClicks)
	}
}

// With a click model deployed, its prediction replaces the prior in the
// price, and a model that's undeployed takes the bidder back to priors.
func TestBidWithModel(t *testing.T) {
	b := newBidder(t, ModeBid)
	src := filepath.Join("..", "model", "testdata", "click-fixture")
	dst := filepath.Join(b.Models.Dir, model.Family, "click-0001")
	os.MkdirAll(dst, 0o755)
	for _, f := range []string{"model.json", "weights.bin", "golden.jsonl"} {
		raw, err := os.ReadFile(filepath.Join(src, f))
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dst, f), raw, 0o644)
	}
	os.WriteFile(filepath.Join(dst, "report.json"), []byte(`{"passed": true}`), 0o644)
	if res := b.Models.Deploy("click-0001", false); !res.OK {
		t.Fatal(res.Message)
	}
	price := func() float64 {
		w := do(b, "POST", "/openrtb/beta_exchange", banner, "1000")
		var resp openrtb.BidResponse
		if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
			t.Fatalf("no bid: %d", w.Code)
		}
		return resp.SeatBid[0].Bid[0].Price
	}
	// price = CPC x p x 1000 x (1 - margin) x shade
	p := price() / (0.10 * 1000 * 0.8 * 0.7)
	if p <= 0 || p >= 1 || math.Abs(p-0.006) < 1e-4 {
		t.Fatalf("model click rate %v: looks like the prior or nonsense", p)
	}
	b.Models.Deploy("", false)
	if got := price(); got != 0.336 {
		t.Fatalf("after undeploy, want the prior price 0.336, got %v", got)
	}
}

// The postback endpoint checks the MMP's token, logs what it sends, and
// counts each install once.
func TestPostbacks(t *testing.T) {
	b := newBidder(t, ModeBid)
	send := func(token, q string) int {
		r := httptest.NewRequest("GET", "/postback?"+q, nil)
		r.Header.Set(clock.Header, "5000")
		if token != "" {
			r.Header.Set("X-MMP-Token", token)
		}
		w := httptest.NewRecorder()
		b.ServeHTTP(w, r)
		return w.Code
	}
	q := "postback_id=pb1&event_name=install&touch_type=click&click_id=cr1&campaign_id=7"
	if code := send("wrong", q); code != http.StatusForbidden {
		t.Fatalf("wrong token: got %d", code)
	}
	if send("secret", q) != http.StatusNoContent || send("secret", q) != http.StatusNoContent {
		t.Fatal("postback rejected")
	}
	send("secret", "postback_id=pb2&event_name=install&touch_type=view&campaign_id=7")
	send("secret", "postback_id=pb3&event_name=af_purchase&touch_type=click&campaign_id=7")
	s := b.Stats.Snapshot()
	if c := s.Campaigns[7]; c.Installs != 1 || c.ViewInstalls != 1 || s.DupPostbacks != 1 {
		t.Fatalf("installs %d, view installs %d, duplicates %d", c.Installs, c.ViewInstalls, s.DupPostbacks)
	}
}

func deployFixture(t *testing.T, b *Bidder) {
	t.Helper()
	src := filepath.Join("..", "model", "testdata", "click-fixture")
	dst := filepath.Join(b.Models.Dir, model.Family, "click-0001")
	os.MkdirAll(dst, 0o755)
	ents, _ := os.ReadDir(src)
	for _, e := range ents {
		raw, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dst, e.Name()), raw, 0o644)
	}
	os.WriteFile(filepath.Join(dst, "report.json"), []byte(`{"passed": true}`), 0o644)
	if res := b.Models.Deploy("click-0001", false); !res.OK {
		t.Fatal(res.Message)
	}
}

// Each kill switch stops bidding at its level, and logs why (Chapter 12).
func TestKillSwitches(t *testing.T) {
	b := newBidder(t, ModeBid)
	code := func() int { return do(b, "POST", "/openrtb/beta_exchange", banner, "1000").Code }
	if code() != http.StatusOK {
		t.Fatal("no bid before any switch")
	}
	for _, c := range []struct{ kind, key string }{
		{"global", ""}, {"exchange", "beta_exchange"}, {"campaign", "7"},
	} {
		b.Ops.Set(c.kind, c.key, true, "test", 0)
		if got := code(); got != http.StatusNoContent {
			t.Errorf("%s switch on: got %d", c.kind, got)
		}
		b.Ops.Set(c.kind, c.key, false, "test", 0)
		if code() != http.StatusOK {
			t.Errorf("%s switch off: no bid", c.kind)
		}
	}
	s := b.Stats.Snapshot()
	for _, r := range []string{"kill_global", "kill_exchange", "campaign_paused"} {
		if s.Reasons[r] != 1 {
			t.Errorf("reason %s counted %d times", r, s.Reasons[r])
		}
	}
	// which campaign was filtered, and why (Chapter 6)
	if got := s.Campaigns[7].Filtered["paused"]; got != 1 {
		t.Errorf("campaign 7 filtered as paused %d times, want 1", got)
	}
	// bids by the version that priced them (Chapter 11)
	if len(s.BidsByVersion) != 1 || s.BidsByVersion[0].Bids["priors"] != 4 {
		t.Errorf("bids by version: %+v", s.BidsByVersion)
	}
}

// The fallback ladder: the model; with its switch on, the bundle's
// segment click rates; with no bundle, the placement priors.
func TestFallbackLadder(t *testing.T) {
	b := newBidder(t, ModeBid)
	deployFixture(t, b)
	fv := &features.FeatureVector{Strings: map[string]string{"placement": "banner", "country": "bra",
		"exchange": "beta_exchange"}, Numeric: map[string]float64{}}
	ad := map[string]string{"campaign": "7"}
	pr, _ := b.predict(fv, ad, "beta_exchange", b.Ops.Switches())
	if pr.PClickFrom != "model" {
		t.Fatalf("want the model, got %s", pr.PClickFrom)
	}
	b.Ops.Set("model", "", true, "test", 0)
	pr, _ = b.predict(fv, ad, "beta_exchange", b.Ops.Switches())
	m := b.Models.Current()
	if pr.PClickFrom != "fallback_segment" || pr.PClick != m.ClickRates.Lookup("banner", "bra", "beta_exchange") {
		t.Fatalf("model switched off: %+v", pr)
	}
	// a feature switched off is missing for the model
	b.Ops.Set("model", "", false, "test", 0)
	b.Ops.Set("feature", "placement", true, "test", 0)
	pr2, _ := b.predict(fv, ad, "beta_exchange", b.Ops.Switches())
	off := withoutFeatures(fv, map[string]bool{"placement": true})
	if pr2.PClickFrom != "model" || pr2.PClick != m.Predict(off, ad) {
		t.Fatalf("feature switch: %+v", pr2)
	}
	b.Models.Deploy("", false)
	if _, ok := b.predict(fv, ad, "beta_exchange", b.Ops.Switches()); ok {
		t.Fatal("with no bundle, the policy's priors should price")
	}
}

// In the internal auction, the bidder prices every campaign; the bid must
// carry the ad of the campaign that won the auction, not the last one
// priced. The response's ad markup and click URL name the campaign.
func TestAuctionBidsWithTheWinnersAd(t *testing.T) {
	b := newBidder(t, ModeBid)
	b.Book.Load([]campaigns.Campaign{
		{ID: 1, Goal: "CPC", Target: 0.50, DailyBudget: 5, Geos: "BRA"},
		{ID: 2, Goal: "CPC", Target: 0.05, DailyBudget: 5, Geos: "BRA"},
		{ID: 3, Goal: "CPC", Target: 0.02, DailyBudget: 5, Geos: "BRA"}},
		[]campaigns.Creative{{ID: 10, CampaignID: 1, Format: "static"},
			{ID: 20, CampaignID: 2, Format: "static"}, {ID: 30, CampaignID: 3, Format: "static"}})
	b.Policy.SetSelection("auction")
	b.Policy.AuctionExplore = 0
	w := do(b, "POST", "/openrtb/beta_exchange", banner, "1000")
	if w.Code != http.StatusOK {
		t.Fatalf("want a bid, got %d", w.Code)
	}
	body := w.Body.String()
	// the highest target has the most surplus: campaign 1, creative 10,
	// even though campaign 3 was priced last
	if !strings.Contains(body, `"cid":"1"`) || !strings.Contains(body, `"crid":"10"`) {
		t.Fatalf("the bid isn't campaign 1's: %s", body)
	}
	if ad := b.lastAdForTest; ad["campaign"] != "1" || ad["creative"] != "10" {
		t.Fatalf("logged ad %v", ad)
	}
}

// A load test request runs the whole bid path and gets a bid, but leaves
// nothing behind: no ledger row, no profile, no pending bid (Chapter 17).
func TestLoadTestWritesNothing(t *testing.T) {
	b := newBidder(t, ModeBid)
	r := httptest.NewRequest("POST", "/openrtb/beta_exchange", strings.NewReader(banner))
	r.Header.Set(clock.Header, "1000")
	r.Header.Set(LoadTestHeader, "1")
	w := httptest.NewRecorder()
	b.ServeHTTP(w, r)
	if w.Code != http.StatusOK {
		t.Fatalf("want a bid, got %d", w.Code)
	}
	if n := b.Profiles.Devices(); n != 0 {
		t.Fatalf("load test recorded %d devices", n)
	}
	if len(b.pending) != 0 || b.Stats.Snapshot().Total != 0 {
		t.Fatalf("load test left a pending bid or stats")
	}
	if s := b.LoadLat.Summary(); s.Requests != 1 || b.Lat.Summary().Requests != 0 {
		t.Fatalf("timings went to the wrong recorder")
	}
	// a live request is timed by stage, and the stages add up
	do(b, "POST", "/openrtb/beta_exchange", banner, "2000")
	s := b.Lat.Summary()
	sum := 0.0
	for _, st := range s.Stages[:len(s.Stages)-1] {
		sum += st.Mean
	}
	if tot := s.Stages[len(s.Stages)-1].Mean; s.Requests != 1 || sum > tot+1 || sum < tot*0.5 {
		t.Fatalf("stages sum to %v of a %v total", sum, tot)
	}
}
