package deep

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"time"
)

type timing struct {
	P50 float64 `json:"p50_us"`
	P99 float64 `json:"p99_us"`
}

func measure(n int, f func(i int)) timing {
	for i := 0; i < n/10+1; i++ { // warm up
		f(i)
	}
	t := make([]float64, n)
	for i := 0; i < n; i++ {
		t0 := time.Now()
		f(i)
		t[i] = float64(time.Since(t0).Nanoseconds()) / 1e3
	}
	sort.Float64s(t)
	return timing{P50: round(t[n/2]), P99: round(t[n*99/100])}
}

func round(x float64) float64 {
	if x >= 100 {
		return float64(int(x + 0.5))
	}
	return float64(int(x*10+0.5)) / 10
}

// Check loads the bundle adtrain export wrote in dir, checks its golden
// vectors, benchmarks it with reps requests per configuration, and checks the
// versioned vectors. It prints as it goes to w, writes deepcheck.json in dir,
// and returns the same report.
func Check(dir string, reps int, w io.Writer) (map[string]any, error) {
	rep := map[string]any{}
	m, worst, err := Load(filepath.Join(dir, "dcn.json"))
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(w, "DCNv2 %s: %d parameters, loaded in pure Go; golden vectors match PyTorch, largest logit difference %.1e\n",
		m.Version, m.Params(), worst)
	rep["version"], rep["params"], rep["golden_max_diff"] = m.Version, m.Params(), worst

	// one request, one candidate
	reqs := m.Requests
	sc := m.NewScratch()
	out := make([]float64, len(m.Heads))
	one := measure(reps, func(i int) { m.Logits(&reqs[i%len(reqs)], sc, out) })
	fmt.Fprintf(w, "  one candidate, three heads: %.1f us p50, %.1f us p99 (one core)\n", one.P50, one.P99)
	rep["one_candidate"] = one

	// many candidates per request
	fmt.Fprintln(w, "  per request, p50 / p99, the whole network per candidate, and with the request's share of the first layers done once:")
	fmt.Fprintf(w, "    %10s %24s %24s\n", "candidates", "whole network", "request's share once")
	cands := m.Candidates
	rows := map[string]any{}
	for _, n := range []int{1, 10, 50} {
		if n > len(cands) {
			n = len(cands)
		}
		buf := make([][]float64, n)
		for i := range buf {
			buf[i] = make([]float64, len(m.Heads))
		}
		r := rand.New(rand.NewSource(7))
		sets := make([][]Candidate, 64)
		for i := range sets {
			idx := r.Perm(len(cands))[:n]
			for _, j := range idx {
				sets[i] = append(sets[i], cands[j])
			}
		}
		k := reps
		if n >= 50 {
			k = reps / 4
		}
		naive := measure(k, func(i int) { m.ScoreNaive(&reqs[i%len(reqs)], sets[i%len(sets)], sc, buf) })
		split := measure(k, func(i int) { m.ScoreSplit(&reqs[i%len(reqs)], sets[i%len(sets)], sc, buf) })
		fmt.Fprintf(w, "    %10d %13.0f / %-6.0f us %13.0f / %-6.0f us\n", n, naive.P50, naive.P99, split.P50, split.P99)
		rows[fmt.Sprint(n)] = map[string]timing{"per_candidate": naive, "request_share_once": split}
	}
	rep["candidates"] = rows

	// versioned vectors
	apps, err1 := LoadVectors(filepath.Join(dir, "app_vectors.json"))
	states, err2 := LoadVectors(filepath.Join(dir, "device_states.json"))
	if err1 != nil || err2 != nil {
		fmt.Fprintln(w, "  no app vectors or device states (run adtrain pretrain before adtrain export):", err1, err2)
	} else {
		var adv []string
		seen := map[string]bool{}
		for _, c := range cands {
			if !seen[c.AdvBundle] && apps.Row(c.AdvBundle) != nil {
				seen[c.AdvBundle] = true
				adv = append(adv, c.AdvBundle)
			}
		}
		scorer := NewScorer(apps)
		ok := 0
		for i, d := range states.Keys {
			_ = d
			st := State{Version: states.Version, H: states.Data[i*states.Dim : (i+1)*states.Dim]}
			for _, a := range adv {
				if _, good := scorer.Affinity(st, a); good {
					ok++
				}
			}
		}
		fmt.Fprintf(w, "  app vectors %s (%d apps) and %d device states of version %s: %d affinities scored, %d fallbacks\n",
			apps.Version, len(apps.Keys), len(states.Keys), states.Version, ok, scorer.Fallbacks)
		st0 := State{Version: states.Version, H: states.Data[:states.Dim]}
		dot := measure(reps*10, func(i int) { scorer.Affinity(st0, adv[i%len(adv)]) })
		fmt.Fprintf(w, "  one affinity, a %d-dimension dot product and two lookups: %.2f us p50\n", apps.Dim, dot.P50)
		// a rollover: half the device records come from a new model version
		next := apps.Version + "-next"
		mixed := 0
		scorer = NewScorer(apps)
		for i := range states.Keys {
			v := states.Version
			if i%2 == 1 {
				v = next
			}
			st := State{Version: v, H: states.Data[i*states.Dim : (i+1)*states.Dim]}
			if _, good := scorer.Affinity(st, adv[0]); good {
				mixed++
			}
		}
		fmt.Fprintf(w, "  rollover, half the device records from a new version and only the old tables loaded: %d scored, %d refused (missing to the trees)\n",
			mixed, scorer.Fallbacks)
		// load the new version's tables alongside (here, the same vectors relabeled)
		both := NewScorer(apps, NewVectors(next, apps.Dim, apps.Keys, apps.Data))
		mixedBoth := 0
		for i := range states.Keys {
			v := states.Version
			if i%2 == 1 {
				v = next
			}
			if _, good := both.Affinity(State{Version: v, H: states.Data[i*states.Dim : (i+1)*states.Dim]}, adv[0]); good {
				mixedBoth++
			}
		}
		fmt.Fprintf(w, "  with both versions' tables loaded: %d scored, %d refused\n", mixedBoth, both.Fallbacks)
		rep["vectors"] = map[string]any{"version": apps.Version, "apps": len(apps.Keys), "device_states": len(states.Keys),
			"affinities": ok, "dot_p50_us": dot.P50, "rollover_scored": mixed, "rollover_refused": scorer.Fallbacks,
			"rollover_both_loaded_scored": mixedBoth}
	}

	// dot products at the chapter's scale: a campaign matrix of 64 dimensions
	r := rand.New(rand.NewSource(7))
	for _, n := range []int{2000, 50000} {
		data := make([]float32, n*64)
		for i := range data {
			data[i] = float32(r.NormFloat64())
		}
		keys := make([]string, n)
		mat := NewVectors("synthetic", 64, keys, data)
		q := data[:64]
		scores := make([]float32, n)
		top := make([]int, 0, 10)
		k := reps
		if n > 10000 {
			k = reps / 10
		}
		t := measure(k, func(int) { top = TopK(mat, q, 10, scores, top) })
		fmt.Fprintf(w, "  %d campaign vectors of 64 dimensions, every one scored and the top 10 kept: %.0f us p50, %.0f us p99\n", n, t.P50, t.P99)
		rep[fmt.Sprintf("topk_%d", n)] = t
	}
	rep["checked"] = time.Now().UTC().Format(time.RFC3339)
	b, _ := json.MarshalIndent(rep, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "deepcheck.json"), b, 0o644); err != nil {
		return rep, err
	}
	fmt.Fprintf(w, "  wrote %s\n", filepath.Join(dir, "deepcheck.json"))
	return rep, nil
}
