// Package ledger writes the bidder's logs: the bid ledger, one JSON line
// per request with the FeatureVector and every term of the decision, and
// the notice log, one line per win, loss, or billing notice.
//
// Writes go through a buffered channel to one goroutine per stream, so a
// slow disk never adds latency to the bid path. If the buffer fills,
// records are dropped and counted rather than blocking.
//
// Files are gzip-compressed JSON lines, partitioned by simulated day:
//
//	<dir>/<stream>/day=0003/part-0000.jsonl.gz
//
// A part is closed when it reaches 250,000 records, when the day changes,
// after two seconds without traffic, and on shutdown. Closed parts are
// complete files. Open parts are flushed every half second, so a training
// job can read everything up to the last flush while the bidder runs: a
// gzip reader sees the records, then an early end of the stream.
package ledger

import (
	"bufio"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"time"
)

const (
	maxPerPart = 250_000
	idleClose  = 2 * time.Second
)

type item struct {
	day int64
	b   []byte
}

// Writer writes one stream, such as "bids" or "notices".
type Writer struct {
	dir     string
	ch      chan item
	dropped atomic.Int64
	written atomic.Int64
	wg      sync.WaitGroup
}

// New starts a writer for <dir>/<stream>.
func New(dir, stream string) (*Writer, error) {
	d := filepath.Join(dir, stream)
	if err := os.MkdirAll(d, 0o755); err != nil {
		return nil, err
	}
	w := &Writer{dir: d, ch: make(chan item, 1<<16)}
	w.wg.Add(1)
	go w.run()
	return w, nil
}

// Log queues one record for a day without blocking.
func (w *Writer) Log(day int64, v any) {
	b, err := json.Marshal(v)
	if err != nil {
		w.dropped.Add(1)
		return
	}
	select {
	case w.ch <- item{day, b}:
	default:
		w.dropped.Add(1)
	}
}

func (w *Writer) Written() int64 { return w.written.Load() }
func (w *Writer) Dropped() int64 { return w.dropped.Load() }

// NextDay is the day after the latest day in the stream's directory, or 0
// for an empty ledger, so a new run can continue the market instead of
// replaying days the ledger already has.
func (w *Writer) NextDay() int64 {
	ents, _ := os.ReadDir(w.dir)
	next := int64(0)
	for _, e := range ents {
		var d int64
		if _, err := fmt.Sscanf(e.Name(), "day=%d", &d); err == nil && e.IsDir() && d+1 > next {
			next = d + 1
		}
	}
	return next
}

// Close flushes everything queued and closes the files.
func (w *Writer) Close() {
	close(w.ch)
	w.wg.Wait()
}

type part struct {
	f     *os.File
	gz    *gzip.Writer
	bw    *bufio.Writer
	n     int
	dirty bool // records written since the last flush
}

// flush makes everything written so far readable, without closing.
func (p *part) flush() {
	if p.dirty {
		p.bw.Flush()
		p.gz.Flush()
		p.dirty = false
	}
}

func (p *part) close() {
	p.bw.Flush()
	p.gz.Close()
	p.f.Close()
}

func (w *Writer) run() {
	defer w.wg.Done()
	open := map[int64]*part{}
	next := map[int64]int{} // next part number per day
	last := time.Now()
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()
	closeAll := func() {
		for d, p := range open {
			p.close()
			delete(open, d)
		}
	}
	defer closeAll()
	for {
		select {
		case it, ok := <-w.ch:
			if !ok {
				return
			}
			last = time.Now()
			p := open[it.day]
			if p == nil {
				var err error
				if p, err = w.newPart(it.day, next); err != nil {
					w.dropped.Add(1)
					continue
				}
				open[it.day] = p
			}
			p.bw.Write(it.b)
			p.bw.WriteByte('\n')
			p.n++
			p.dirty = true
			w.written.Add(1)
			if p.n >= maxPerPart {
				p.close()
				delete(open, it.day)
			}
			// Only the latest days stay open; older days are done.
			for d, op := range open {
				if d < it.day-1 {
					op.close()
					delete(open, d)
				}
			}
		case <-tick.C:
			if len(open) > 0 && time.Since(last) > idleClose {
				closeAll()
			}
			for _, p := range open {
				p.flush()
			}
		}
	}
}

func (w *Writer) newPart(day int64, next map[int64]int) (*part, error) {
	dayDir := filepath.Join(w.dir, fmt.Sprintf("day=%04d", day))
	if err := os.MkdirAll(dayDir, 0o755); err != nil {
		return nil, err
	}
	for {
		name := filepath.Join(dayDir, fmt.Sprintf("part-%04d.jsonl.gz", next[day]))
		next[day]++
		f, err := os.OpenFile(name, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if os.IsExist(err) {
			continue // left by an earlier run; never overwrite
		}
		if err != nil {
			return nil, err
		}
		gz := gzip.NewWriter(f)
		return &part{f: f, gz: gz, bw: bufio.NewWriterSize(gz, 1<<20)}, nil
	}
}
