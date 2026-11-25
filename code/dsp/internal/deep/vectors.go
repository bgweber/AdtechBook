package deep

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
)

// Vectors is a versioned table of vectors in one contiguous row-major
// slice: app vectors, or device states. Scoring every row is one
// matrix-vector product over memory the CPU can stream.
type Vectors struct {
	Version string
	Dim     int
	Keys    []string
	Data    []float32
	index   map[string]int
}

type vectorsFile struct {
	Version string    `json:"version"`
	Dim     int       `json:"dim"`
	Apps    []string  `json:"apps"`
	Devices []string  `json:"devices"`
	Data    []float32 `json:"data"`
}

// LoadVectors reads app vectors or device states exported by adtrain.
func LoadVectors(path string) (*Vectors, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var f vectorsFile
	if err := json.Unmarshal(b, &f); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	keys := f.Apps
	if keys == nil {
		keys = f.Devices
	}
	if f.Version == "" || f.Dim <= 0 || len(f.Data) != len(keys)*f.Dim {
		return nil, fmt.Errorf("%s: version %q, %d keys of %d dimensions, %d values", path, f.Version, len(keys), f.Dim, len(f.Data))
	}
	return NewVectors(f.Version, f.Dim, keys, f.Data), nil
}

func NewVectors(version string, dim int, keys []string, data []float32) *Vectors {
	v := &Vectors{Version: version, Dim: dim, Keys: keys, Data: data, index: make(map[string]int, len(keys))}
	for i, k := range keys {
		v.index[k] = i
	}
	return v
}

// Row returns the vector for a key, or nil.
func (v *Vectors) Row(key string) []float32 {
	i, ok := v.index[key]
	if !ok {
		return nil
	}
	return v.Data[i*v.Dim : (i+1)*v.Dim]
}

// Dot is a dot product, unrolled four ways.
func Dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	k := 0
	for ; k+4 <= len(a); k += 4 {
		s0 += a[k] * b[k]
		s1 += a[k+1] * b[k+1]
		s2 += a[k+2] * b[k+2]
		s3 += a[k+3] * b[k+3]
	}
	for ; k < len(a); k++ {
		s0 += a[k] * b[k]
	}
	return s0 + s1 + s2 + s3
}

// State is a device's state from the profile, with the version of the model
// that computed it.
type State struct {
	Version string
	H       []float32
}

// Affinity is the device's lift for an app: the dot product of its state
// and the app's vector. It refuses, and returns false, when the state and
// the vectors come from different model versions: those scores look normal
// and are wrong. The trees then treat the affinity as missing.
func (v *Vectors) Affinity(s State, app string) (float32, bool) {
	if s.H == nil || s.Version != v.Version {
		return 0, false
	}
	e := v.Row(app)
	if e == nil || len(e) != len(s.H) {
		return 0, false
	}
	return Dot(s.H, e), true
}

// Scorer keeps the vector tables of every live version, so that during a
// rollover each device record is scored with the tables of its own version.
type Scorer struct {
	tables map[string]*Vectors
	// Fallbacks counts the records whose version has no table.
	Fallbacks int
}

func NewScorer(tables ...*Vectors) *Scorer {
	s := &Scorer{tables: map[string]*Vectors{}}
	for _, t := range tables {
		s.tables[t.Version] = t
	}
	return s
}

// Affinity scores a device record with the table of its version, or falls
// back to missing.
func (s *Scorer) Affinity(st State, app string) (float32, bool) {
	t, ok := s.tables[st.Version]
	if !ok {
		s.Fallbacks++
		return 0, false
	}
	return t.Affinity(st, app)
}

// Unload drops a version once nothing refers to it.
func (s *Scorer) Unload(version string) { delete(s.tables, version) }

// TopK scores every row of a matrix against a vector and returns the k best
// rows, best first, with a small partial selection rather than a full sort.
// scores is a buffer of at least len(keys) entries, reused per worker.
func TopK(m *Vectors, q []float32, k int, scores []float32, top []int) []int {
	n := len(m.Keys)
	D := m.Dim
	for i := 0; i < n; i++ {
		scores[i] = Dot(m.Data[i*D:(i+1)*D], q)
	}
	top = top[:0]
	for i := 0; i < n; i++ {
		if len(top) < k {
			top = append(top, i)
			for j := len(top) - 1; j > 0 && scores[top[j]] > scores[top[j-1]]; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
			continue
		}
		if scores[i] <= scores[top[k-1]] {
			continue
		}
		top[k-1] = i
		for j := k - 1; j > 0 && scores[top[j]] > scores[top[j-1]]; j-- {
			top[j], top[j-1] = top[j-1], top[j]
		}
	}
	return top
}

// topKSort is the reference TopK for tests: a full sort.
func topKSort(m *Vectors, q []float32, k int) []int {
	idx := make([]int, len(m.Keys))
	sc := make([]float32, len(m.Keys))
	for i := range idx {
		idx[i] = i
		sc[i] = Dot(m.Data[i*m.Dim:(i+1)*m.Dim], q)
	}
	sort.SliceStable(idx, func(a, b int) bool { return sc[idx[a]] > sc[idx[b]] })
	if k > len(idx) {
		k = len(idx)
	}
	return idx[:k]
}
