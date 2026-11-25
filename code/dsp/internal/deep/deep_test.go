package deep

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const tiny = "testdata/dcn_tiny.json"

func TestLoadChecksGoldenVectors(t *testing.T) {
	m, worst, err := Load(tiny)
	if err != nil {
		t.Fatal(err)
	}
	if worst > 1e-5 {
		t.Fatalf("golden vectors differ by %g", worst)
	}
	if !strings.HasPrefix(m.Version, "dcn-") || len(m.Heads) != 3 {
		t.Fatalf("version %q, heads %v", m.Version, m.Heads)
	}
}

func TestLoadRefusesAModelThatDoesntMatch(t *testing.T) {
	b, err := os.ReadFile(tiny)
	if err != nil {
		t.Fatal(err)
	}
	var s map[string]any
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	// change one weight of a head: the golden vectors no longer match
	h := s["head_weights"].(map[string]any)["click"].(map[string]any)["b"].(map[string]any)
	h["data"].([]any)[0] = h["data"].([]any)[0].(float64) + 0.01
	p := filepath.Join(t.TempDir(), "bad.json")
	b, _ = json.Marshal(s)
	if err := os.WriteFile(p, b, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Load(p); err == nil || !strings.Contains(err.Error(), "golden") {
		t.Fatalf("loaded a model that doesn't match its golden vectors: %v", err)
	}
}

func TestSplitScoringMatchesTheWholeNetwork(t *testing.T) {
	m, _, err := Load(tiny)
	if err != nil {
		t.Fatal(err)
	}
	sc := m.NewScratch()
	cands := m.Candidates
	a := make([][]float64, len(cands))
	b := make([][]float64, len(cands))
	for i := range a {
		a[i], b[i] = make([]float64, 3), make([]float64, 3)
	}
	for r := range m.Requests {
		m.ScoreNaive(&m.Requests[r], cands, sc, a)
		m.ScoreSplit(&m.Requests[r], cands, sc, b)
		for i := range a {
			for k := range a[i] {
				if math.Abs(a[i][k]-b[i][k]) > 1e-4 {
					t.Fatalf("request %d, candidate %d, head %d: %g against %g", r, i, k, b[i][k], a[i][k])
				}
			}
		}
	}
	// a candidate scored alone matches the request with its fields set
	in := m.Requests[0]
	c := cands[0]
	ids := map[string]string{}
	for k, v := range in.IDs {
		ids[k] = v
	}
	ids["campaign"], ids["creative"], ids["creative_format"], ids["goal"] = c.Campaign, c.Creative, c.CreativeFormat, c.Goal
	in.IDs = ids
	in.Apps = map[string]string{"publisher_app": in.Apps["publisher_app"], "adv_bundle": c.AdvBundle}
	out := make([]float64, 3)
	m.Logits(&in, sc, out)
	m.ScoreSplit(&m.Requests[0], cands[:1], sc, b[:1])
	for k := range out {
		if math.Abs(out[k]-b[0][k]) > 1e-4 {
			t.Fatalf("head %d: %g against %g", k, b[0][k], out[k])
		}
	}
}

func TestScoringDoesntAllocate(t *testing.T) {
	m, _, err := Load(tiny)
	if err != nil {
		t.Fatal(err)
	}
	sc := m.NewScratch()
	out := make([][]float64, len(m.Candidates))
	for i := range out {
		out[i] = make([]float64, 3)
	}
	n := testing.AllocsPerRun(50, func() { m.ScoreSplit(&m.Requests[0], m.Candidates, sc, out) })
	if n > 0 {
		t.Fatalf("%v allocations per request", n)
	}
}

func TestAffinityRefusesMismatchedVersions(t *testing.T) {
	apps := NewVectors("v42", 2, []string{"com.a", "com.b"}, []float32{1, 0, 0, 1})
	st := State{Version: "v42", H: []float32{2, 3}}
	if v, ok := apps.Affinity(st, "com.b"); !ok || v != 3 {
		t.Fatalf("affinity %v %v", v, ok)
	}
	if _, ok := apps.Affinity(State{Version: "v41", H: []float32{2, 3}}, "com.b"); ok {
		t.Fatal("scored a device state from another version")
	}
	if _, ok := apps.Affinity(st, "com.unknown"); ok {
		t.Fatal("scored an app with no vector")
	}
	// during a rollover both versions are loaded, and each record uses its own
	old := NewVectors("v41", 2, []string{"com.a", "com.b"}, []float32{0, 1, 1, 0})
	s := NewScorer(apps, old)
	if v, ok := s.Affinity(State{Version: "v41", H: []float32{2, 3}}, "com.b"); !ok || v != 2 {
		t.Fatalf("old version: %v %v", v, ok)
	}
	s.Unload("v41")
	if _, ok := s.Affinity(State{Version: "v41", H: []float32{2, 3}}, "com.b"); ok || s.Fallbacks != 1 {
		t.Fatalf("scored a record whose version was unloaded; fallbacks %d", s.Fallbacks)
	}
}

func TestTopKMatchesAFullSort(t *testing.T) {
	data := make([]float32, 300*4)
	for i := range data {
		data[i] = float32(math.Sin(float64(i) * 1.7))
	}
	m := NewVectors("x", 4, make([]string, 300), data)
	q := []float32{0.3, -1, 0.5, 2}
	scores := make([]float32, 300)
	got := TopK(m, q, 10, scores, nil)
	want := topKSort(m, q, 10)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("top %d: %v, want %v", i, got, want)
		}
	}
	if n := len(TopK(NewVectors("x", 4, make([]string, 3), data[:12]), q, 10, scores, nil)); n != 3 {
		t.Fatalf("top 10 of 3: %d", n)
	}
}

func BenchmarkOneCandidate(b *testing.B) {
	m, _, err := Load(tiny)
	if err != nil {
		b.Fatal(err)
	}
	sc := m.NewScratch()
	out := make([]float64, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Logits(&m.Requests[i%len(m.Requests)], sc, out)
	}
}
