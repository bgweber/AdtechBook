package profile

import (
	"compress/gzip"
	"encoding/gob"
	"io"
	"os"
	"path/filepath"
)

// saved is one device's profile on disk.
type saved struct {
	IFA                               string
	FirstSeenMS, LastSeenMS, Requests int64
	LastImpMS, ImpDay, ImpsToday      int64
	Installs, Purchases               int64
	Apps                              map[string]int64
	Recent                            []Visit
}

// Save writes the store to path, replacing the old file only once the new
// one is complete. In walk, the bidder saves on shutdown and every few
// minutes, so a restart doesn't wipe device history: features built from
// it would otherwise drop to zero for every device, a training-serving
// skew of the bidder's own making (Chapter 15). Production stores keep
// profiles out of process instead (Chapter 17).
func (st *Store) Save(path string) (int, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return 0, err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return 0, err
	}
	zw := gzip.NewWriter(f)
	enc := gob.NewEncoder(zw)
	n := 0
	for i := range st.s {
		sh := &st.s[i]
		sh.mu.Lock()
		for ifa, p := range sh.m {
			err = enc.Encode(saved{ifa, p.FirstSeenMS, p.LastSeenMS, p.Requests,
				p.LastImpMS, p.ImpDay, p.ImpsToday, p.Installs, p.Purchases, p.apps, p.recent})
			if err != nil {
				break
			}
			n++
		}
		sh.mu.Unlock()
		if err != nil {
			f.Close()
			return 0, err
		}
	}
	if err = zw.Close(); err == nil {
		err = f.Close()
	}
	if err != nil {
		return 0, err
	}
	return n, os.Rename(tmp, path)
}

// Load reads profiles written by Save. A missing file is an empty store.
func (st *Store) Load(path string) (int, error) {
	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return 0, nil
	} else if err != nil {
		return 0, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return 0, err
	}
	dec := gob.NewDecoder(zr)
	n := 0
	for {
		var s saved
		if err := dec.Decode(&s); err != nil {
			if err == io.EOF {
				return n, nil
			}
			return n, err
		}
		if s.Apps == nil {
			s.Apps = map[string]int64{}
		}
		sh := st.shard(s.IFA)
		sh.mu.Lock()
		sh.m[s.IFA] = &Profile{FirstSeenMS: s.FirstSeenMS, LastSeenMS: s.LastSeenMS,
			Requests: s.Requests, LastImpMS: s.LastImpMS, ImpDay: s.ImpDay,
			ImpsToday: s.ImpsToday, Installs: s.Installs, Purchases: s.Purchases,
			apps: s.Apps, recent: s.Recent}
		sh.mu.Unlock()
		n++
	}
}
