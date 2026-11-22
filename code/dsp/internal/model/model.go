// Package model serves the click model of Chapter 9 in the bidder.
//
// The training code (code/training) publishes a model artifact: a
// metadata file, model.json, that says how to turn a FeatureVector and
// the chosen ad into hashed tokens, and a weights file, weights.bin, with
// the nonzero weights as sorted (4-byte bucket, 4-byte float) pairs. The
// bidder reads both and computes nothing from its own configuration, so
// training and serving can't drift apart by accident.
//
// Every artifact also carries golden.jsonl: examples with the prediction
// the training code made for them. The bidder refuses to deploy a model
// whose predictions it can't reproduce (Chapter 11).
package model

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// Spec is how to build tokens. It comes from the artifact, never from
// the bidder's flags.
type Spec struct {
	Strings        []string             `json:"strings"`         // one token each; missing -> name_null
	NumericBuckets map[string][]float64 `json:"numeric_buckets"` // name_b{i}; missing -> name_missing
	Sets           map[string]int       `json:"sets"`            // many-hot, at most N members
	Crosses        [][2]string          `json:"crosses"`         // a_va*b_vb when both are present
	Hash           struct {
		Fn   string `json:"fn"`
		Seed uint32 `json:"seed"`
	} `json:"hash"`
	NumBuckets uint32 `json:"num_buckets"`
}

// Model is a trained logistic regression with hashed tokens.
type Model struct {
	Name           string             `json:"model"`
	Version        string             `json:"version"`
	FeatureSchema  int                `json:"feature_schema_version"`
	Bias           float64            `json:"bias"`
	DownsampleRate float64            `json:"downsample_rate"`
	NonzeroWeights int                `json:"nonzero_weights"`
	Spec           Spec               `json:"spec"`
	Weights        map[uint32]float32 `json:"-"`
	// Rates are the bundle's click-to-install rates, for pricing install
	// campaigns (Chapter 10); nil if the bundle has none.
	Rates *InstallRates `json:"-"`
	// ClickRates are the bundle's segment click rates, the fallback when
	// the model can't be used (Chapter 12); nil if the bundle has none.
	ClickRates *ClickRates `json:"-"`
}

// Tokens turns one FeatureVector and the chosen ad into tokens, in the
// same order as the training code. The order doesn't change the score,
// but it makes the two easy to compare in a test.
func (s *Spec) Tokens(fv *features.FeatureVector, ad map[string]string) []string {
	val := func(k string) (string, bool) {
		if v, ok := ad[k]; ok && v != "" {
			return v, true
		}
		v, ok := fv.Strings[k]
		return v, ok && v != ""
	}
	var toks []string
	for _, k := range s.Strings {
		v, ok := val(k)
		if !ok {
			v = "null"
		}
		toks = append(toks, k+"_"+v)
	}
	for _, k := range sortedKeys(s.NumericBuckets) {
		x, ok := fv.Numeric[k]
		if !ok {
			toks = append(toks, k+"_missing")
			continue
		}
		edges := s.NumericBuckets[k]
		i := sort.Search(len(edges), func(i int) bool { return edges[i] > x })
		toks = append(toks, fmt.Sprintf("%s_b%d", k, i))
	}
	for _, k := range sortedKeys(s.Sets) {
		members := append([]string(nil), fv.Sets[k]...)
		sort.Strings(members)
		if len(members) > s.Sets[k] {
			members = members[:s.Sets[k]]
		}
		for _, m := range members {
			toks = append(toks, k+"_"+m)
		}
	}
	for _, c := range s.Crosses {
		va, oka := val(c[0])
		vb, okb := val(c[1])
		if oka && okb {
			toks = append(toks, c[0]+"_"+va+"*"+c[1]+"_"+vb)
		}
	}
	return toks
}

// Buckets hashes tokens into distinct bucket indexes. Two tokens that
// collide count once, as in training.
func (s *Spec) Buckets(toks []string) []uint32 {
	seen := make(map[uint32]bool, len(toks))
	out := make([]uint32, 0, len(toks))
	for _, t := range toks {
		b := Murmur3([]byte(t), s.Hash.Seed) % s.NumBuckets
		if !seen[b] {
			seen[b] = true
			out = append(out, b)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Predict returns the click probability, corrected for negative
// downsampling.
func (m *Model) Predict(fv *features.FeatureVector, ad map[string]string) float64 {
	z := m.Bias
	for _, b := range m.Spec.Buckets(m.Spec.Tokens(fv, ad)) {
		z += float64(m.Weights[b])
	}
	p := 1 / (1 + math.Exp(-z))
	if w := m.DownsampleRate; w > 0 && w < 1 {
		p = p / (p + (1-p)/w)
	}
	return p
}

// Load reads a model artifact from its directory.
func Load(dir string) (*Model, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	m := &Model{}
	if err := json.Unmarshal(raw, m); err != nil {
		return nil, fmt.Errorf("model.json: %w", err)
	}
	if m.Spec.Hash.Fn != "murmur3_32" {
		return nil, fmt.Errorf("unsupported hash %q", m.Spec.Hash.Fn)
	}
	if m.Spec.NumBuckets == 0 {
		return nil, fmt.Errorf("num_buckets is 0")
	}
	if m.FeatureSchema > features.SchemaVersion {
		return nil, fmt.Errorf("model needs FeatureVector schema %d, bidder has %d",
			m.FeatureSchema, features.SchemaVersion)
	}
	f, err := os.Open(filepath.Join(dir, "weights.bin"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m.Weights = make(map[uint32]float32, m.NonzeroWeights)
	r := bufio.NewReader(f)
	var pair [8]byte
	for {
		if _, err := io.ReadFull(r, pair[:]); err == io.EOF {
			break
		} else if err != nil {
			return nil, fmt.Errorf("weights.bin: %w", err)
		}
		b := binary.LittleEndian.Uint32(pair[:4])
		m.Weights[b] = math.Float32frombits(binary.LittleEndian.Uint32(pair[4:]))
	}
	if len(m.Weights) != m.NonzeroWeights {
		return nil, fmt.Errorf("weights.bin has %d weights, model.json says %d",
			len(m.Weights), m.NonzeroWeights)
	}
	if m.Rates, err = LoadRates(dir); err != nil {
		return nil, fmt.Errorf("install_rates.json: %w", err)
	}
	if m.ClickRates, err = LoadClickRates(dir); err != nil {
		return nil, fmt.Errorf("click_rates.json: %w", err)
	}
	return m, nil
}

// Golden is one example from golden.jsonl.
type Golden struct {
	Features features.FeatureVector `json:"features"`
	Ad       map[string]string      `json:"ad"`
	P        float64                `json:"p"`
}

// ParityTolerance is how far a bidder prediction may be from the
// training code's. Both add the same float32 weights, so only the order
// of the additions differs.
const ParityTolerance = 1e-9

// CheckGolden predicts every golden example and returns the number of
// examples and the largest difference from the training code.
func (m *Model) CheckGolden(dir string) (int, float64, error) {
	f, err := os.Open(filepath.Join(dir, "golden.jsonl"))
	if err != nil {
		return 0, 0, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 16<<20)
	n, worst := 0, 0.0
	for sc.Scan() {
		var g Golden
		if err := json.Unmarshal(sc.Bytes(), &g); err != nil {
			return n, worst, fmt.Errorf("golden.jsonl line %d: %w", n+1, err)
		}
		worst = math.Max(worst, math.Abs(m.Predict(&g.Features, g.Ad)-g.P))
		n++
	}
	if err := sc.Err(); err != nil {
		return n, worst, err
	}
	if n == 0 {
		return 0, 0, fmt.Errorf("golden.jsonl is empty")
	}
	return n, worst, nil
}

func sortedKeys[V any](m map[string]V) []string {
	ks := make([]string, 0, len(m))
	for k := range m {
		ks = append(ks, k)
	}
	sort.Strings(ks)
	return ks
}
