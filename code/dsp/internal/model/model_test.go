package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// The same tokens hashed by Python's mmh3.hash(token, seed, signed=False).
func TestMurmur3MatchesPython(t *testing.T) {
	want := map[string]uint32{
		"": 0, "a": 1009084850, "ab": 2613040991, "abc": 3017643002, "abcd": 1139631978,
		"placement_banner": 3073373577, "country_ind": 2981217691,
		"publisher_app_com.example.gemgarden*campaign_3": 2898163447,
		"héllo_wörld": 3367439918,
	}
	for s, h := range want {
		if got := Murmur3([]byte(s), 0); got != h {
			t.Errorf("Murmur3(%q, 0) = %d, want %d", s, got, h)
		}
	}
	seeded := map[string]uint32{"": 142593372, "a": 3001393763, "ab": 3610054215, "abc": 1313807976}
	for s, h := range seeded {
		if got := Murmur3([]byte(s), 42); got != h {
			t.Errorf("Murmur3(%q, 42) = %d, want %d", s, got, h)
		}
	}
}

// testdata comes from the training code: python -m adtrain.fixture.
func TestTokensMatchPython(t *testing.T) {
	m, err := Load(filepath.Join("testdata", "click-fixture"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join("testdata", "tokens.json"))
	if err != nil {
		t.Fatal(err)
	}
	var rows []struct {
		Features features.FeatureVector `json:"features"`
		Ad       map[string]string      `json:"ad"`
		Tokens   []string               `json:"tokens"`
		Buckets  []uint32               `json:"buckets"`
	}
	if err := json.Unmarshal(raw, &rows); err != nil {
		t.Fatal(err)
	}
	for i, r := range rows {
		toks := m.Spec.Tokens(&r.Features, r.Ad)
		if !reflect.DeepEqual(toks, r.Tokens) {
			t.Errorf("example %d tokens:\n got %v\nwant %v", i, toks, r.Tokens)
		}
		if b := m.Spec.Buckets(toks); !reflect.DeepEqual(b, r.Buckets) {
			t.Errorf("example %d buckets differ", i)
		}
	}
}

func TestGoldenParity(t *testing.T) {
	dir := filepath.Join("testdata", "click-fixture")
	m, err := Load(dir)
	if err != nil {
		t.Fatal(err)
	}
	if m.DownsampleRate != 0.5 {
		t.Fatalf("fixture should use downsampling, got rate %v", m.DownsampleRate)
	}
	n, worst, err := m.CheckGolden(dir)
	if err != nil {
		t.Fatal(err)
	}
	if n < 10 || worst > ParityTolerance {
		t.Fatalf("parity: %d examples, max diff %g", n, worst)
	}
}

func copyDir(t *testing.T, src, dst string) {
	t.Helper()
	os.MkdirAll(dst, 0o755)
	ents, _ := os.ReadDir(src)
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644)
	}
}

func TestDeployGates(t *testing.T) {
	dir := t.TempDir()
	r := &Registry{Dir: dir}
	v := filepath.Join(dir, Family, "click-0001")
	copyDir(t, filepath.Join("testdata", "click-fixture"), v)
	write := func(passed bool) {
		os.WriteFile(filepath.Join(v, "report.json"), []byte(`{"passed": `+
			map[bool]string{true: "true", false: "false"}[passed]+
			`, "checks": [{"name": "stability", "pass": `+
			map[bool]string{true: "true", false: "false"}[passed]+`}]}`), 0o644)
	}

	write(false)
	if res := r.Deploy("click-0001", false); res.OK || r.Current() != nil {
		t.Fatalf("a model that failed its checks deployed without force: %+v", res)
	}
	if res := r.Deploy("click-0001", true); !res.OK || r.Current() == nil {
		t.Fatalf("forced deploy failed: %+v", res)
	}

	// a restart comes back with the deployed model
	r2 := &Registry{Dir: dir}
	if got, err := r2.LoadDeployed(); err != nil || got != "click-0001" || r2.Current() == nil {
		t.Fatalf("LoadDeployed = %q, %v", got, err)
	}

	// parity can't be forced: change one golden prediction
	golden := filepath.Join(v, "golden.jsonl")
	b, _ := os.ReadFile(golden)
	var g map[string]any
	lines := splitLines(b)
	json.Unmarshal(lines[0], &g)
	g["p"] = g["p"].(float64) + 0.01
	first, _ := json.Marshal(g)
	lines[0] = first
	os.WriteFile(golden, joinLines(lines), 0o644)
	write(true)
	if res := r.Deploy("click-0001", true); res.OK {
		t.Fatalf("deployed a model that fails parity: %+v", res)
	}

	if res := r.Deploy("", false); !res.OK || r.Current() != nil {
		t.Fatalf("undeploy failed: %+v", res)
	}
	if res := r.Deploy("../etc", true); res.OK {
		t.Fatal("deployed a path outside the models directory")
	}
}

func splitLines(b []byte) [][]byte {
	var out [][]byte
	start := 0
	for i, c := range b {
		if c == '\n' {
			out = append(out, b[start:i])
			start = i + 1
		}
	}
	if start < len(b) {
		out = append(out, b[start:])
	}
	return out
}

func joinLines(ls [][]byte) []byte {
	var out []byte
	for _, l := range ls {
		out = append(out, l...)
		out = append(out, '\n')
	}
	return out
}

func TestInstallRates(t *testing.T) {
	m, err := Load(filepath.Join("testdata", "click-fixture"))
	if err != nil {
		t.Fatal(err)
	}
	r := m.Rates
	if r == nil || r.K != 200 {
		t.Fatalf("fixture bundle should carry install rates, got %+v", r)
	}
	if v, from := r.Lookup("2", "ind", "banner"); from != "segment" || v != r.Segments["2|ind|banner"] {
		t.Errorf("segment lookup: %v from %s", v, from)
	}
	if v, from := r.Lookup("2", "xyz", "banner"); from != "campaign" || v != r.Campaigns["2"] {
		t.Errorf("campaign fallback: %v from %s", v, from)
	}
	if v, from := r.Lookup("99", "ind", "banner"); from != "overall" || v != r.Overall {
		t.Errorf("overall fallback: %v from %s", v, from)
	}
}
