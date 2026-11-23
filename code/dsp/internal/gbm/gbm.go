// Package gbm serves the gradient-boosted conversion models of Chapter 15.
//
// The training code (code/training, `adtrain gbm`) publishes a bundle:
// model.json, with the feature list and the vocabulary that turns strings
// into category codes, and one file per model (click.json,
// install_click.json, install_view.json) with LightGBM's own dump of its
// trees. The bidder walks the trees itself, in Go, with no cgo and no
// LightGBM library, and follows LightGBM's rules for missing values and
// category splits exactly. golden.jsonl holds examples with the
// predictions LightGBM made for them, and the bidder refuses a bundle
// whose predictions it can't reproduce.
package gbm

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/bgweber/adtechbook/code/dsp/internal/features"
)

// adKeys are the inputs that come from the chosen ad, not the request.
var adKeys = map[string]bool{"campaign": true, "creative": true,
	"creative_format": true, "adv_app": true, "goal": true}

// Spec says how to turn a FeatureVector and an ad into model inputs.
type Spec struct {
	Version       string                    `json:"version"`
	Categorical   []string                  `json:"categorical"`
	Numeric       []string                  `json:"numeric"`
	ZeroIfMissing []string                  `json:"zero_if_missing"`
	Other         string                    `json:"other"`
	Vocab         map[string]map[string]int `json:"vocab"`
	Targets       []string                  `json:"targets"`
}

// Bundle is a loaded set of models that share one Spec.
type Bundle struct {
	Spec   Spec
	Models map[string]*Ensemble // by target: click, install_click, ...
	Values *Values              // D7 revenue per install, if the bundle has it
	zero   map[string]bool
}

// Values is the expected D7 revenue per install, by advertiser app,
// shrunk toward the overall value (Chapter 21). ROAS pricing uses it when
// there are too few purchases for a purchase model.
type Values struct {
	Overall  float64            `json:"overall"`
	ByAdvApp map[string]float64 `json:"by_adv_app"`
}

// ValuePerInstall is the expected D7 revenue per install for an app.
func (b *Bundle) ValuePerInstall(advApp string) (float64, bool) {
	if b == nil || b.Values == nil {
		return 0, false
	}
	if v, ok := b.Values.ByAdvApp[advApp]; ok {
		return v, true
	}
	return b.Values.Overall, b.Values.Overall > 0
}

// Inputs builds one row of model inputs: category codes, then numbers.
// Strings the vocabulary doesn't know share the "other" code, and missing
// numbers are NaN, except the ones logged only when nonzero.
func (b *Bundle) Inputs(fv *features.FeatureVector, ad map[string]string) []float64 {
	x := make([]float64, 0, len(b.Spec.Categorical)+len(b.Spec.Numeric))
	for _, k := range b.Spec.Categorical {
		var v string
		if adKeys[k] {
			v = ad[k]
		} else {
			v = fv.Strings[k]
		}
		codes := b.Spec.Vocab[k]
		c, ok := codes[v]
		if !ok {
			c = codes[b.Spec.Other]
		}
		x = append(x, float64(c))
	}
	for _, k := range b.Spec.Numeric {
		if v, ok := fv.Numeric[k]; ok {
			x = append(x, v)
		} else if b.zero[k] {
			x = append(x, 0)
		} else {
			x = append(x, math.NaN())
		}
	}
	return x
}

// Predict returns one model's probability, and false if the bundle has no
// model for the target.
func (b *Bundle) Predict(target string, x []float64) (float64, bool) {
	m, ok := b.Models[target]
	if !ok {
		return 0, false
	}
	return m.Predict(x), true
}

// PredictBatch is Predict for many rows of inputs at once.
func (b *Bundle) PredictBatch(target string, xs [][]float64, out []float64) bool {
	m, ok := b.Models[target]
	if !ok {
		return false
	}
	m.PredictBatch(xs, out)
	return true
}

// Load reads a bundle directory.
func Load(dir string) (*Bundle, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "model.json"))
	if err != nil {
		return nil, err
	}
	b := &Bundle{Models: map[string]*Ensemble{}, zero: map[string]bool{}}
	if err := json.Unmarshal(raw, &b.Spec); err != nil {
		return nil, fmt.Errorf("model.json: %w", err)
	}
	for _, k := range b.Spec.ZeroIfMissing {
		b.zero[k] = true
	}
	for _, k := range b.Spec.Categorical {
		if _, ok := b.Spec.Vocab[k][b.Spec.Other]; !ok {
			return nil, fmt.Errorf("model.json: no %q code for %s", b.Spec.Other, k)
		}
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "values.json")); err == nil {
		var v Values
		if err := json.Unmarshal(raw, &v); err != nil {
			return nil, fmt.Errorf("values.json: %w", err)
		}
		b.Values = &v
	}
	for _, t := range b.Spec.Targets {
		e, err := LoadEnsemble(filepath.Join(dir, t+".json"))
		if err != nil {
			return nil, fmt.Errorf("%s: %w", t, err)
		}
		if e.NumFeatures != len(b.Spec.Categorical)+len(b.Spec.Numeric) {
			return nil, fmt.Errorf("%s: %d features, spec has %d", t,
				e.NumFeatures, len(b.Spec.Categorical)+len(b.Spec.Numeric))
		}
		b.Models[t] = e
	}
	return b, nil
}

// CheckGolden predicts every golden example and returns how many there
// were and the largest difference from LightGBM's prediction.
func (b *Bundle) CheckGolden(dir string) (int, float64, error) {
	rows, err := goldenRows(dir)
	if err != nil {
		return 0, 0, err
	}
	worst := 0.0
	for _, g := range rows {
		x := b.Inputs(&g.Features, g.Ad)
		for t, want := range g.Predictions {
			got, ok := b.Predict(t, x)
			if !ok {
				return 0, worst, fmt.Errorf("golden example for unknown model %q", t)
			}
			worst = math.Max(worst, math.Abs(got-want))
		}
	}
	return len(rows), worst, nil
}

// goldenRow is one line of golden.jsonl.
type goldenRow struct {
	Features    features.FeatureVector `json:"features"`
	Ad          map[string]string      `json:"ad"`
	Predictions map[string]float64     `json:"predictions"`
}

func goldenRows(dir string) ([]goldenRow, error) {
	f, err := os.Open(filepath.Join(dir, "golden.jsonl"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<22)
	var rows []goldenRow
	for sc.Scan() {
		var g goldenRow
		if err := json.Unmarshal(sc.Bytes(), &g); err != nil {
			return rows, err
		}
		rows = append(rows, g)
	}
	return rows, sc.Err()
}

// Ensemble is one LightGBM binary model: the sum of its trees, through a
// sigmoid.
type Ensemble struct {
	NumFeatures int
	Sigmoid     float64
	Trees       []Tree
}

// Tree is one tree, flattened: node i's children are Left[i] and Right[i],
// and a negative child -k-1 is leaf k.
type Tree struct {
	Feature     []int
	Threshold   []float64
	Categories  [][]uint64 // bitset of left categories; nil for numerical splits
	DefaultLeft []bool
	Missing     []byte // 'N' none, 'Z' zero, 'n' NaN
	Left, Right []int
	Leaf        []float64
}

// Predict is the model's probability for one row of inputs.
func (e *Ensemble) Predict(x []float64) float64 {
	s := 0.0
	for i := range e.Trees {
		s += e.Trees[i].value(x)
	}
	return 1 / (1 + math.Exp(-e.Sigmoid*s))
}

// PredictBatch fills out[r] with the probability for xs[r]. It walks
// tree by tree rather than row by row, so each tree stays in cache while
// every candidate passes through it; the sums are the same as Predict's.
func (e *Ensemble) PredictBatch(xs [][]float64, out []float64) {
	for r := range xs {
		out[r] = 0
	}
	for i := range e.Trees {
		t := &e.Trees[i]
		for r, x := range xs {
			out[r] += t.value(x)
		}
	}
	for r := range xs {
		out[r] = 1 / (1 + math.Exp(-e.Sigmoid*out[r]))
	}
}

const zeroThreshold = 1e-35

func (t *Tree) value(x []float64) float64 {
	if len(t.Feature) == 0 {
		return t.Leaf[0]
	}
	i := 0
	for {
		next := t.next(i, x[t.Feature[i]])
		if next < 0 {
			return t.Leaf[-next-1]
		}
		i = next
	}
}

// next follows LightGBM's own decision rules (NumericalDecision and
// CategoricalDecision in its tree.h), so missing values go where they
// went in training.
func (t *Tree) next(i int, v float64) int {
	if cats := t.Categories[i]; cats != nil {
		var c int
		if math.IsNaN(v) {
			if t.Missing[i] == 'n' {
				return t.Right[i]
			}
			c = 0
		} else {
			c = int(v)
		}
		if c >= 0 && c/64 < len(cats) && cats[c/64]&(1<<(c%64)) != 0 {
			return t.Left[i]
		}
		return t.Right[i]
	}
	if math.IsNaN(v) && t.Missing[i] != 'n' {
		v = 0
	}
	if (t.Missing[i] == 'Z' && math.Abs(v) <= zeroThreshold) ||
		(t.Missing[i] == 'n' && math.IsNaN(v)) {
		if t.DefaultLeft[i] {
			return t.Left[i]
		}
		return t.Right[i]
	}
	if v <= t.Threshold[i] {
		return t.Left[i]
	}
	return t.Right[i]
}

// LoadEnsemble reads LightGBM's dump_model() JSON.
func LoadEnsemble(path string) (*Ensemble, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var d struct {
		MaxFeatureIdx int    `json:"max_feature_idx"`
		Objective     string `json:"objective"`
		NumClass      int    `json:"num_class"`
		TreeInfo      []struct {
			TreeStructure json.RawMessage `json:"tree_structure"`
		} `json:"tree_info"`
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		return nil, err
	}
	if d.NumClass != 1 || !strings.HasPrefix(d.Objective, "binary") {
		return nil, fmt.Errorf("only binary models are supported, not %q", d.Objective)
	}
	e := &Ensemble{NumFeatures: d.MaxFeatureIdx + 1, Sigmoid: 1}
	if i := strings.Index(d.Objective, "sigmoid:"); i >= 0 {
		if s, err := strconv.ParseFloat(strings.Fields(d.Objective[i+8:])[0], 64); err == nil {
			e.Sigmoid = s
		}
	}
	for _, ti := range d.TreeInfo {
		var n node
		if err := json.Unmarshal(ti.TreeStructure, &n); err != nil {
			return nil, err
		}
		t := Tree{}
		if n.LeafValue != nil && n.Left == nil {
			t.Leaf = []float64{*n.LeafValue}
		} else if _, err := t.add(&n); err != nil {
			return nil, err
		}
		e.Trees = append(e.Trees, t)
	}
	return e, nil
}

type node struct {
	SplitFeature *int            `json:"split_feature"`
	Threshold    json.RawMessage `json:"threshold"`
	DecisionType string          `json:"decision_type"`
	DefaultLeft  bool            `json:"default_left"`
	MissingType  string          `json:"missing_type"`
	LeafValue    *float64        `json:"leaf_value"`
	Left         *node           `json:"left_child"`
	Right        *node           `json:"right_child"`
}

// add appends a node and its subtree, and returns its index, or -k-1 for
// leaf k.
func (t *Tree) add(n *node) (int, error) {
	if n.Left == nil {
		if n.LeafValue == nil {
			return 0, fmt.Errorf("leaf without a value")
		}
		t.Leaf = append(t.Leaf, *n.LeafValue)
		return -len(t.Leaf), nil
	}
	i := len(t.Feature)
	t.Feature = append(t.Feature, *n.SplitFeature)
	t.DefaultLeft = append(t.DefaultLeft, n.DefaultLeft)
	t.Left = append(t.Left, 0)
	t.Right = append(t.Right, 0)
	switch n.MissingType {
	case "Zero":
		t.Missing = append(t.Missing, 'Z')
	case "NaN":
		t.Missing = append(t.Missing, 'n')
	default:
		t.Missing = append(t.Missing, 'N')
	}
	switch n.DecisionType {
	case "<=":
		var th float64
		if err := json.Unmarshal(n.Threshold, &th); err != nil {
			return 0, fmt.Errorf("threshold: %w", err)
		}
		t.Threshold = append(t.Threshold, th)
		t.Categories = append(t.Categories, nil)
	case "==":
		var s string
		if err := json.Unmarshal(n.Threshold, &s); err != nil {
			return 0, fmt.Errorf("category threshold: %w", err)
		}
		// A bitset, as LightGBM keeps them: a map lookup per split was
		// half the evaluator's time (Chapter 17).
		var cats []uint64
		for _, p := range strings.Split(s, "||") {
			c, err := strconv.Atoi(p)
			if err != nil || c < 0 {
				return 0, fmt.Errorf("category %q: %v", p, err)
			}
			for len(cats) <= c/64 {
				cats = append(cats, 0)
			}
			cats[c/64] |= 1 << (c % 64)
		}
		t.Threshold = append(t.Threshold, 0)
		t.Categories = append(t.Categories, cats)
	default:
		return 0, fmt.Errorf("decision type %q", n.DecisionType)
	}
	l, err := t.add(n.Left)
	if err != nil {
		return 0, err
	}
	r, err := t.add(n.Right)
	if err != nil {
		return 0, err
	}
	t.Left[i], t.Right[i] = l, r
	return i, nil
}
