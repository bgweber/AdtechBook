// Package deep serves the deep models of Part IV in the bidder (Chapter 28):
// the multi-task DCNv2 of Chapter 26, run in pure Go from the bundle adtrain
// exports, and the versioned app vectors and device states of Chapter 27,
// scored with dot products.
//
// The reference bidder has no dependencies and no cgo, so instead of ONNX
// Runtime it runs the network itself, as it runs LightGBM. The bundle's
// golden vectors, raw inputs with PyTorch's outputs, are checked when the
// model loads, and a model that doesn't match doesn't load.
package deep

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

type matrix struct {
	Rows int       `json:"rows"`
	Cols int       `json:"cols"`
	Data []float32 `json:"data"`
}

type layer struct {
	W matrix `json:"w"`
	B matrix `json:"b"`
}

// Input is one impression as the bidder sees it: IDs as strings, the
// publisher and advertised apps, the device's recent apps, and raw numbers.
// A nil number is missing.
type Input struct {
	IDs        map[string]string   `json:"ids"`
	Apps       map[string]string   `json:"apps"`
	RecentApps []string            `json:"recent_apps"`
	Numeric    map[string]*float64 `json:"numeric"`
	Hour       *float64            `json:"hour_of_day"`
}

// Candidate is what varies between the candidates of one request.
type Candidate struct {
	Campaign       string `json:"campaign"`
	Creative       string `json:"creative"`
	CreativeFormat string `json:"creative_format"`
	Goal           string `json:"goal"`
	AdvBundle      string `json:"adv_bundle"`
}

type golden struct {
	Input  Input     `json:"input"`
	Logits []float64 `json:"logits"`
}

type spec struct {
	Format      string                    `json:"format"`
	Version     string                    `json:"version"`
	Dim         int                       `json:"dim"`
	Fields      []string                  `json:"fields"`
	AppFields   []string                  `json:"app_fields"`
	Numeric     []string                  `json:"numeric"`
	Mean        []float32                 `json:"numeric_mean"`
	SD          []float32                 `json:"numeric_sd"`
	MaxRecent   int                       `json:"max_recent_apps"`
	Pad         int                       `json:"pad"`
	OOV         int                       `json:"oov"`
	Vocabs      map[string]map[string]int `json:"vocabs"`
	Heads       []string                  `json:"heads"`
	Embeddings  map[string]matrix         `json:"embeddings"`
	AppEmb      matrix                    `json:"app_embedding"`
	Cross       []layer                   `json:"cross"`
	MLP         []layer                   `json:"mlp"`
	HeadWeights map[string]layer          `json:"head_weights"`
	Golden      []golden                  `json:"golden"`
	Tolerance   float64                   `json:"tolerance"`
	Requests    []Input                   `json:"requests"`
	Candidates  []Candidate               `json:"candidates"`
}

// DCN is a loaded multi-task DCNv2.
type DCN struct {
	Version string
	Heads   []string
	// a sample of the test day's requests and the dataset's candidates, for
	// the benchmark
	Requests   []Input
	Candidates []Candidate

	dim, d, hidden int
	fields         []string
	appFields      []string
	numeric        []string
	vocab          []map[string]int
	appVocab       map[string]int
	emb            []matrix
	app            matrix
	mean, sd       []float32
	maxRecent, oov int
	cross, mlp     []layer
	heads          []layer
	// the slots of x0 that depend on the candidate
	candField map[string]int
	candApp   int
	slots     [][2]int
}

// Load reads an exported bundle and checks its golden vectors. It returns
// the largest difference from PyTorch's logits.
func Load(path string) (*DCN, float64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var s spec
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	if s.Format != "adtrain-dcnv2-1" {
		return nil, 0, fmt.Errorf("%s: unknown format %q", path, s.Format)
	}
	m := &DCN{Version: s.Version, Heads: s.Heads, Requests: s.Requests, Candidates: s.Candidates,
		dim: s.Dim, fields: s.Fields, appFields: s.AppFields, numeric: s.Numeric, appVocab: s.Vocabs["app"],
		app: s.AppEmb, mean: s.Mean, sd: s.SD, maxRecent: s.MaxRecent, oov: s.OOV, cross: s.Cross, mlp: s.MLP,
		candField: map[string]int{}, candApp: -1}
	for i, f := range s.Fields {
		m.vocab = append(m.vocab, s.Vocabs[f])
		m.emb = append(m.emb, s.Embeddings[f])
		switch f {
		case "campaign", "creative", "creative_format", "goal":
			m.candField[f] = i
		}
	}
	for i, f := range s.AppFields {
		if f == "adv_bundle" {
			m.candApp = i
		}
	}
	for _, h := range s.Heads {
		m.heads = append(m.heads, s.HeadWeights[h])
	}
	m.d = s.Dim*(len(s.Fields)+len(s.AppFields)+1) + len(s.Numeric) + 2
	m.slots = m.candSlots()
	if len(s.MLP) > 0 {
		m.hidden = s.MLP[len(s.MLP)-1].W.Rows
	}
	if err := m.check(); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	// golden vectors: the model loads only if Go matches PyTorch
	sc := m.NewScratch()
	out := make([]float64, len(m.heads))
	worst := 0.0
	for i := range s.Golden {
		g := &s.Golden[i]
		m.Logits(&g.Input, sc, out)
		for k := range out {
			if d := math.Abs(out[k] - g.Logits[k]); d > worst || math.IsNaN(d) {
				worst = d
			}
		}
	}
	if len(s.Golden) == 0 {
		return nil, 0, fmt.Errorf("%s: no golden vectors", path)
	}
	if !(worst <= s.Tolerance) {
		return nil, worst, fmt.Errorf("%s: golden vectors differ from PyTorch by %.2g, more than %.2g", path, worst, s.Tolerance)
	}
	return m, worst, nil
}

func (m *DCN) check() error {
	if len(m.mean) != len(m.numeric)+2 || len(m.sd) != len(m.mean) {
		return fmt.Errorf("numeric transform: %d means for %d numbers", len(m.mean), len(m.numeric)+2)
	}
	for i, e := range m.emb {
		if e.Cols != m.dim || len(e.Data) != e.Rows*e.Cols {
			return fmt.Errorf("embedding %s: shape %dx%d", m.fields[i], e.Rows, e.Cols)
		}
	}
	in := m.d
	for i, l := range m.cross {
		if l.W.Rows != m.d || l.W.Cols != m.d || len(l.B.Data) != m.d {
			return fmt.Errorf("cross layer %d: shape %dx%d", i, l.W.Rows, l.W.Cols)
		}
	}
	for i, l := range m.mlp {
		if l.W.Cols != in || len(l.B.Data) != l.W.Rows {
			return fmt.Errorf("mlp layer %d: shape %dx%d after %d", i, l.W.Rows, l.W.Cols, in)
		}
		in = l.W.Rows
	}
	for i, l := range m.heads {
		if l.W.Cols != m.d+m.hidden {
			return fmt.Errorf("head %s: %d inputs", m.Heads[i], l.W.Cols)
		}
	}
	return nil
}

// Scratch holds one worker's buffers, so scoring doesn't allocate.
type Scratch struct {
	x0, xl, t, z []float32
	h            [][]float32
	// per request: the products of the first layers with the request's part
	// of x0
	cross0, mlp0 []float32
	c1, m1       []float32
}

func (m *DCN) NewScratch() *Scratch {
	s := &Scratch{x0: make([]float32, m.d), xl: make([]float32, m.d), t: make([]float32, m.d),
		z: make([]float32, m.d+m.hidden), cross0: make([]float32, m.d), c1: make([]float32, m.d)}
	for _, l := range m.mlp {
		s.h = append(s.h, make([]float32, l.W.Rows))
	}
	if len(m.mlp) > 0 {
		s.mlp0 = make([]float32, m.mlp[0].W.Rows)
		s.m1 = make([]float32, m.mlp[0].W.Rows)
	}
	return s
}

func (m *DCN) id(field int, v string) int {
	if i, ok := m.vocab[field][v]; ok {
		return i
	}
	return m.oov
}

func (m *DCN) appID(v string) int {
	if i, ok := m.appVocab[v]; ok {
		return i
	}
	return m.oov
}

func row(e matrix, i int) []float32 { return e.Data[i*e.Cols : (i+1)*e.Cols] }

// encode writes x0: one embedding per ID field, the two apps from the
// shared table, the mean of the recent apps, and the transformed numbers.
func (m *DCN) encode(in *Input, x0 []float32) {
	D := m.dim
	for i, f := range m.fields {
		copy(x0[i*D:(i+1)*D], row(m.emb[i], m.id(i, in.IDs[f])))
	}
	off := len(m.fields) * D
	for j, f := range m.appFields {
		copy(x0[off+j*D:off+(j+1)*D], row(m.app, m.appID(in.Apps[f])))
	}
	off += len(m.appFields) * D
	pool := x0[off : off+D]
	for k := range pool {
		pool[k] = 0
	}
	n := len(in.RecentApps)
	if n > m.maxRecent {
		n = m.maxRecent // the training width
	}
	for _, a := range in.RecentApps[:n] {
		for k, v := range row(m.app, m.appID(a)) {
			pool[k] += v
		}
	}
	if n > 0 {
		for k := range pool {
			pool[k] /= float32(n)
		}
	}
	off += D
	for k, name := range m.numeric {
		v := 0.0
		if p := in.Numeric[name]; p != nil && !math.IsNaN(*p) {
			v = *p
		}
		x0[off+k] = float32(math.Log1p(math.Max(v, 0)))
	}
	h := 0.0
	if in.Hour != nil && !math.IsNaN(*in.Hour) {
		h = *in.Hour
	}
	nn := len(m.numeric)
	x0[off+nn] = float32(math.Sin(2 * math.Pi * h / 24))
	x0[off+nn+1] = float32(math.Cos(2 * math.Pi * h / 24))
	for k := 0; k < nn+2; k++ {
		x0[off+k] = (x0[off+k] - m.mean[k]) / m.sd[k]
	}
}

// matvec computes y = W x + b, unrolled four ways so the compiler can keep
// the accumulators in registers.
func matvec(W matrix, b []float32, x, y []float32) {
	n := W.Cols
	for r := 0; r < W.Rows; r++ {
		w := W.Data[r*n : (r+1)*n]
		var s0, s1, s2, s3 float32
		k := 0
		for ; k+4 <= n; k += 4 {
			s0 += w[k] * x[k]
			s1 += w[k+1] * x[k+1]
			s2 += w[k+2] * x[k+2]
			s3 += w[k+3] * x[k+3]
		}
		for ; k < n; k++ {
			s0 += w[k] * x[k]
		}
		y[r] = s0 + s1 + s2 + s3 + b[r]
	}
}

// forward runs the network on x0; first, if not nil, holds the first cross
// layer's and first MLP layer's products W x0 + b already computed.
func (m *DCN) forward(sc *Scratch, cross0, mlp0 []float32, out []float64) {
	x0, xl, t := sc.x0, sc.xl, sc.t
	copy(xl, x0)
	for i, l := range m.cross { // DCNv2: x_{l+1} = x0 * (W x_l + b) + x_l
		if i == 0 && cross0 != nil {
			copy(t, cross0)
		} else {
			matvec(l.W, l.B.Data, xl, t)
		}
		for k := range xl {
			xl[k] = x0[k]*t[k] + xl[k]
		}
	}
	in := x0
	for i, l := range m.mlp {
		h := sc.h[i]
		if i == 0 && mlp0 != nil {
			copy(h, mlp0)
		} else {
			matvec(l.W, l.B.Data, in, h)
		}
		for k := range h {
			if h[k] < 0 {
				h[k] = 0
			}
		}
		in = h
	}
	z := sc.z
	copy(z, xl)
	copy(z[m.d:], in)
	for k, l := range m.heads {
		var s float32
		for j, w := range l.W.Data {
			s += w * z[j]
		}
		out[k] = float64(s + l.B.Data[0])
	}
}

// Logits scores one input: one logit per head.
func (m *DCN) Logits(in *Input, sc *Scratch, out []float64) {
	m.encode(in, sc.x0)
	m.forward(sc, nil, nil, out)
}

// candSlots lists the x0 positions a candidate fills.
func (m *DCN) candSlots() [][2]int {
	var s [][2]int
	for _, i := range m.candField {
		s = append(s, [2]int{i * m.dim, (i + 1) * m.dim})
	}
	if m.candApp >= 0 {
		o := len(m.fields)*m.dim + m.candApp*m.dim
		s = append(s, [2]int{o, o + m.dim})
	}
	return s
}

func (m *DCN) setCandidate(c *Candidate, x0 []float32) {
	D := m.dim
	for f, i := range m.candField {
		var v string
		switch f {
		case "campaign":
			v = c.Campaign
		case "creative":
			v = c.Creative
		case "creative_format":
			v = c.CreativeFormat
		case "goal":
			v = c.Goal
		}
		copy(x0[i*D:(i+1)*D], row(m.emb[i], m.id(i, v)))
	}
	if m.candApp >= 0 {
		o := len(m.fields)*D + m.candApp*D
		copy(x0[o:o+D], row(m.app, m.appID(c.AdvBundle)))
	}
}

// ScoreNaive scores every candidate of a request with the whole network.
func (m *DCN) ScoreNaive(req *Input, cands []Candidate, sc *Scratch, out [][]float64) {
	m.encode(req, sc.x0)
	for i := range cands {
		m.setCandidate(&cands[i], sc.x0)
		m.forward(sc, nil, nil, out[i])
	}
}

// ScoreSplit scores every candidate of a request, doing the request's share
// of the first layers once: W x0 is the sum of the products with the
// request's slots and with the candidate's slots, and only the candidate's
// slots change.
func (m *DCN) ScoreSplit(req *Input, cands []Candidate, sc *Scratch, out [][]float64) {
	x0 := sc.x0
	m.encode(req, x0)
	slots := m.slots
	for _, s := range slots { // the candidate's slots, empty for the request's share
		for k := s[0]; k < s[1]; k++ {
			x0[k] = 0
		}
	}
	// the request's share, once
	var c0, h0 []float32
	if len(m.cross) > 0 {
		c0 = sc.cross0
		matvec(m.cross[0].W, m.cross[0].B.Data, x0, c0)
	}
	if len(m.mlp) > 0 {
		h0 = sc.mlp0
		matvec(m.mlp[0].W, m.mlp[0].B.Data, x0, h0)
	}
	cross0, mlp0 := sc.c1, sc.m1
	for i := range cands {
		m.setCandidate(&cands[i], x0)
		copy(cross0, c0)
		copy(mlp0, h0)
		for _, s := range slots { // add the candidate's share
			addCols(m.cross[0].W, x0, s, cross0)
			addCols(m.mlp[0].W, x0, s, mlp0)
		}
		m.forward(sc, cross0, mlp0, out[i])
	}
}

// addCols adds W[:, a:b] x[a:b] to y.
func addCols(W matrix, x []float32, s [2]int, y []float32) {
	n := W.Cols
	for r := 0; r < W.Rows; r++ {
		w := W.Data[r*n : (r+1)*n]
		var acc float32
		for k := s[0]; k < s[1]; k++ {
			acc += w[k] * x[k]
		}
		y[r] += acc
	}
}

// Params is the number of weights, for the report.
func (m *DCN) Params() int {
	n := len(m.app.Data)
	for _, e := range m.emb {
		n += len(e.Data)
	}
	for _, ls := range [][]layer{m.cross, m.mlp, m.heads} {
		for _, l := range ls {
			n += len(l.W.Data) + len(l.B.Data)
		}
	}
	return n
}
