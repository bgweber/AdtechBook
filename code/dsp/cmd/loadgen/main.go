// Command loadgen is the bidder's load test (Chapter 17).
//
// It replays real bid requests, the raw sample the bidder logs (Chapter 8),
// at a series of fixed rates, and measures what an exchange would see:
// how long each answer took, and how many missed the deadline. Each
// request carries the X-Load-Test header, so the bidder runs the whole bid
// path but writes nothing. After each rate it reads the bidder's own stage
// timings, so a slow total comes with where the time went.
//
// The load is open loop: requests go out on a schedule whether or not
// earlier ones have answered, as an exchange's do. A closed loop, which
// waits for each answer before sending the next, slows down with the
// bidder and hides the queueing that makes the tail long.
//
//	go run ./cmd/loadgen -raw out/ledger/raw -rates 100,200,400,800
package main

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/latency"
)

type rawRecord struct {
	TsMS     int64           `json:"ts_ms"`
	Exchange string          `json:"exchange"`
	Request  json.RawMessage `json:"request"`
}

// Step is one rate's results.
type Step struct {
	Rate      int     `json:"rate"`     // requests per second asked for
	Achieved  float64 `json:"achieved"` // sent per second
	Sent      int64   `json:"sent"`
	Bids      int64   `json:"bids"`
	NoBids    int64   `json:"nobids"`
	Late      int64   `json:"late"`    // answered after the deadline, or not at all
	Errors    int64   `json:"errors"`  // anything but a 200 or 204
	Dropped   int64   `json:"dropped"` // couldn't be sent: every connection busy
	LateShare float64 `json:"late_share"`
	P50       float64 `json:"p50_ms"`
	P90       float64 `json:"p90_ms"`
	P99       float64 `json:"p99_ms"`
	Max       float64 `json:"max_ms"`
	// Server is the bidder's own stage timings for this step.
	Server *latency.Summary `json:"server,omitempty"`
}

// Run is one load test.
type Run struct {
	Label    string    `json:"label"`
	Started  time.Time `json:"started"`
	Bidder   string    `json:"bidder_version"`
	Requests int       `json:"requests_loaded"`
	TmaxMS   int       `json:"tmax_ms"`
	CPUs     int       `json:"client_cpus"`
	Steps    []Step    `json:"steps"`
	// The auction's candidate cap during the run, and what it gave up on
	// the audited requests.
	AuctionCap int             `json:"auction_cap"`
	CapAudit   json.RawMessage `json:"cap_audit,omitempty"`
}

func main() {
	rawDir := flag.String("raw", "out/ledger/raw", "the raw request log to replay")
	days := flag.Int("days", 3, "replay the last this many days of the log")
	maxReq := flag.Int("max", 20000, "load at most this many requests")
	bidURL := flag.String("bidder", "http://localhost:8080", "the bidder's bid port")
	adminURL := flag.String("admin", "http://localhost:8081", "the bidder's console, for its stage timings")
	rates := flag.String("rates", "50,100,200,400,800", "requests per second, one step each")
	seconds := flag.Float64("seconds", 15, "how long each step runs")
	tmax := flag.Duration("tmax", 100*time.Millisecond, "the deadline: an answer after it is a timeout")
	conns := flag.Int("conns", 256, "most requests in flight at once")
	out := flag.String("out", "out/ops/loadtest.json", "append the results to this file; empty for none")
	label := flag.String("label", "", "a name for this run, such as before or after")
	flag.Parse()

	reqs, err := load(*rawDir, *days, *maxReq)
	if err != nil {
		log.Fatal(err)
	}
	if len(reqs) == 0 {
		log.Fatalf("no requests in %s: run the simulator first", *rawDir)
	}
	// Replayed on the day after the log's last, at the same time of day:
	// no budget is spent yet, so every campaign is eligible, as at the
	// start of a day, the busiest case for the auction.
	last := int64(0)
	for _, r := range reqs {
		last = max(last, r.tsMS/dayMS)
	}
	for i := range reqs {
		reqs[i].tsMS = (last+1)*dayMS + reqs[i].tsMS%dayMS
	}
	if n := campaignCount(*adminURL); n == 0 {
		log.Fatalf("the bidder has no campaigns: run the simulator first, which loads them " +
			"(they're kept until the bidder restarts)")
	}
	client := &http.Client{Timeout: *tmax, Transport: &http.Transport{
		MaxIdleConnsPerHost: *conns, MaxConnsPerHost: *conns}}
	run := Run{Label: *label, Started: time.Now().UTC(), Requests: len(reqs),
		TmaxMS: int(tmax.Milliseconds()), CPUs: runtime.NumCPU()}
	fmt.Printf("replaying %d requests from %s\n", len(reqs), *rawDir)
	fmt.Printf("%6s %8s %7s %7s %7s %7s %8s %8s %8s   server p50/p99 ms, top stage\n",
		"rate", "achieved", "bids", "late", "errors", "dropped", "p50 ms", "p99 ms", "max ms")
	next := 0
	for _, f := range strings.Split(*rates, ",") {
		rate, err := strconv.Atoi(strings.TrimSpace(f))
		if err != nil || rate <= 0 {
			log.Fatalf("bad rate %q", f)
		}
		post(*adminURL + "/api/latency/reset")
		s, version := step(client, *bidURL, reqs, &next, rate, *seconds, *tmax, *conns)
		if version != "" {
			run.Bidder = version
		}
		s.Server = summary(*adminURL)
		run.Steps = append(run.Steps, s)
		srv := ""
		if s.Server != nil && len(s.Server.Stages) > 0 {
			tot := s.Server.Stages[latency.Total]
			top := s.Server.Stages[0]
			for _, st := range s.Server.Stages[:latency.Total] {
				if st.Mean > top.Mean {
					top = st
				}
			}
			srv = fmt.Sprintf("%.2f/%.2f, %s %.0f%%", tot.P50/1000, tot.P99/1000, top.Stage, 100*top.Share)
		}
		fmt.Printf("%6d %8.0f %7d %7d %7d %7d %8.2f %8.2f %8.1f   %s\n", rate, s.Achieved, s.Bids,
			s.Late, s.Errors, s.Dropped, s.P50, s.P99, s.Max, srv)
	}
	if v := policyState(*adminURL); v != nil {
		run.AuctionCap, run.CapAudit = v.Cap, v.Audit
		if v.Cap > 0 {
			fmt.Printf("auction cap %d; audit: %s\n", v.Cap, v.Audit)
		}
	}
	if *out != "" {
		if err := appendRun(*out, run); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("results appended to %s\n", *out)
	}
}

const dayMS = 24 * 60 * 60 * 1000

type request struct {
	tsMS     int64
	exchange string
	body     []byte
}

// load reads the raw log's newest days.
func load(dir string, days, most int) ([]request, error) {
	dayDirs, err := filepath.Glob(filepath.Join(dir, "day=*"))
	if err != nil {
		return nil, err
	}
	sort.Strings(dayDirs)
	if len(dayDirs) > days {
		dayDirs = dayDirs[len(dayDirs)-days:]
	}
	var out []request
	for _, d := range dayDirs {
		files, _ := filepath.Glob(filepath.Join(d, "*.jsonl*"))
		sort.Strings(files)
		for _, f := range files {
			if err := readFile(f, func(r rawRecord) bool {
				out = append(out, request{r.TsMS, r.Exchange, r.Request})
				return len(out) < most
			}); err != nil {
				return out, err
			}
			if len(out) >= most {
				return out, nil
			}
		}
	}
	return out, nil
}

func readFile(path string, each func(rawRecord) bool) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var rd io.Reader = f
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer gz.Close()
		rd = gz
	}
	sc := bufio.NewScanner(rd)
	sc.Buffer(make([]byte, 1<<20), 1<<22)
	for sc.Scan() {
		var r rawRecord
		if json.Unmarshal(sc.Bytes(), &r) != nil || len(r.Request) == 0 {
			continue
		}
		if !each(r) {
			return nil
		}
	}
	// a file the bidder is still writing ends mid-stream: keep what was read
	return nil
}

// step sends requests at a fixed rate for a while.
func step(client *http.Client, base string, reqs []request, next *int, rate int,
	seconds float64, tmax time.Duration, conns int) (Step, string) {
	s := Step{Rate: rate}
	var mu sync.Mutex
	var lat []float64
	var version atomic.Value
	slots := make(chan struct{}, conns)
	var wg sync.WaitGroup
	interval := time.Second / time.Duration(rate)
	n := int(seconds * float64(rate))
	start := time.Now()
	for i := 0; i < n; i++ {
		// open loop: the i-th request leaves at its scheduled time
		if d := time.Until(start.Add(time.Duration(i) * interval)); d > 0 {
			time.Sleep(d)
		}
		select {
		case slots <- struct{}{}:
		default:
			s.Dropped++
			continue
		}
		r := reqs[*next%len(reqs)]
		*next++
		s.Sent++
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-slots }()
			t0 := time.Now()
			req, _ := http.NewRequest("POST", base+"/openrtb/"+r.exchange, bytes.NewReader(r.body))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("X-Load-Test", "1")
			req.Header.Set("X-Adsim-Time", strconv.FormatInt(r.tsMS, 10))
			resp, err := client.Do(req)
			ms := float64(time.Since(t0).Microseconds()) / 1000
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				s.Late++ // the client gave up at the deadline
				return
			}
			io.Copy(io.Discard, resp.Body)
			resp.Body.Close()
			if v := resp.Header.Get("X-Bidder-Version"); v != "" {
				version.Store(v)
			}
			lat = append(lat, ms)
			switch {
			case resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent:
				s.Errors++
			case ms > float64(tmax.Milliseconds()):
				s.Late++
			case resp.StatusCode == http.StatusOK:
				s.Bids++
			default:
				s.NoBids++
			}
		}()
	}
	wg.Wait()
	s.Achieved = float64(s.Sent) / time.Since(start).Seconds()
	if total := s.Sent + s.Dropped; total > 0 {
		s.LateShare = float64(s.Late+s.Dropped) / float64(total)
	}
	sort.Float64s(lat)
	s.P50, s.P90 = latency.Quantile(lat, 0.5), latency.Quantile(lat, 0.9)
	s.P99, s.Max = latency.Quantile(lat, 0.99), latency.Quantile(lat, 1)
	v, _ := version.Load().(string)
	return s, v
}

func campaignCount(admin string) int {
	resp, err := http.Get(admin + "/api/campaigns")
	if err != nil {
		log.Fatalf("can't reach the bidder's console: %v", err)
	}
	defer resp.Body.Close()
	var v []json.RawMessage
	json.NewDecoder(resp.Body).Decode(&v)
	return len(v)
}

type policyView struct {
	Cap   int             `json:"auction_cap"`
	Audit json.RawMessage `json:"cap_audit"`
}

func policyState(admin string) *policyView {
	resp, err := http.Get(admin + "/api/latency")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var v policyView
	if json.NewDecoder(resp.Body).Decode(&v) != nil {
		return nil
	}
	return &v
}

func post(url string) {
	resp, err := http.Post(url, "application/json", strings.NewReader(`{"which":"load"}`))
	if err != nil {
		log.Printf("resetting the bidder's timings: %v (stage timings will be missing)", err)
		return
	}
	resp.Body.Close()
}

func summary(admin string) *latency.Summary {
	resp, err := http.Get(admin + "/api/latency")
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	var v struct {
		Load latency.Summary `json:"load"`
	}
	if json.NewDecoder(resp.Body).Decode(&v) != nil {
		return nil
	}
	return &v.Load
}

// appendRun keeps the last ten runs, so the console can compare a run
// before an optimization with one after.
func appendRun(path string, run Run) error {
	var runs []Run
	if b, err := os.ReadFile(path); err == nil {
		json.Unmarshal(b, &runs)
	}
	runs = append(runs, run)
	if len(runs) > 10 {
		runs = runs[len(runs)-10:]
	}
	b, err := json.MarshalIndent(runs, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
