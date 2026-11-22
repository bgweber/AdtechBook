package ledger

import (
	"bufio"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// A part that's still open is readable up to the last flush, so training
// can run while the bidder is still writing (it reads the records, then
// an unexpected end of the stream).
func TestOpenPartIsReadable(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, "clicks")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	stop := time.After(1500 * time.Millisecond)
	tick := time.NewTicker(100 * time.Millisecond) // steady traffic: never idle
	defer tick.Stop()
	n := 0
loop:
	for {
		select {
		case <-tick.C:
			w.Log(0, map[string]int{"n": n})
			n++
		case <-stop:
			break loop
		}
	}
	f, err := os.Open(filepath.Join(dir, "clicks", "day=0000", "part-0000.jsonl.gz"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	sc := bufio.NewScanner(gz)
	lines := 0
	for sc.Scan() {
		lines++
	}
	if lines < n-6 {
		t.Fatalf("read %d of %d records from the open part", lines, n)
	}
}

func TestNextDay(t *testing.T) {
	dir := t.TempDir()
	w, err := New(dir, "bids")
	if err != nil {
		t.Fatal(err)
	}
	if n := w.NextDay(); n != 0 {
		t.Fatalf("empty ledger: next day %d", n)
	}
	w.Log(3, map[string]int{"x": 1})
	w.Close()
	if n := w.NextDay(); n != 4 {
		t.Fatalf("after day 3: next day %d", n)
	}
}
