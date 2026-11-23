package latency

import (
	"testing"
	"time"
)

func TestStagesAddUpToTheTotal(t *testing.T) {
	tm := Start()
	time.Sleep(2 * time.Millisecond)
	tm.Lap(Parse)
	time.Sleep(3 * time.Millisecond)
	tm.Lap(Price)
	tm.Add(Models, 2*time.Millisecond, Price) // measured inside the price lap
	tm.Done()
	sum := time.Duration(0)
	for s := Stage(0); s < Total; s++ {
		sum += tm.D[s]
	}
	if diff := tm.D[Total] - sum; diff < 0 || diff > time.Millisecond {
		t.Fatalf("stages %v, total %v", sum, tm.D[Total])
	}
	if tm.D[Models] != 2*time.Millisecond || tm.D[Price] >= 3*time.Millisecond {
		t.Fatalf("models %v, price %v", tm.D[Models], tm.D[Price])
	}
}

func TestRecorderPercentilesAndRing(t *testing.T) {
	r := NewRecorder()
	for i := 1; i <= Size+100; i++ {
		tm := &Timer{}
		tm.D[Total] = time.Duration(i) * time.Microsecond
		r.Record(tm)
	}
	s := r.Summary()
	if s.Requests != Size+100 || s.Kept != Size {
		t.Fatalf("requests %d kept %d", s.Requests, s.Kept)
	}
	tot := s.Stages[Total]
	// the ring holds requests 101..Size+100
	if tot.Max != float64(Size+100) || tot.P50 < float64(Size/2) {
		t.Fatalf("total %+v", tot)
	}
	r.Reset()
	if r.Summary().Requests != 0 {
		t.Fatal("reset kept requests")
	}
}
