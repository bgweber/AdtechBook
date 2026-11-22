// Package clock tells the bidder what time it is.
//
// In production that's the wall clock. In the simulator, the market runs
// a simulated day in minutes, and sends the simulated time with each
// request in the X-Adsim-Time header (milliseconds since the start of
// the simulation). Budgets, day boundaries, and logs use that time, so a
// simulated day behaves like a real one.
package clock

import (
	"net/http"
	"strconv"
	"time"
)

// Header carries the simulated time on requests from the simulator.
const Header = "X-Adsim-Time"

// DayMS is one day in milliseconds.
const DayMS = 24 * 60 * 60 * 1000

// Now returns the request's time in milliseconds: simulated time if the
// request carries it, otherwise wall-clock time.
func Now(r *http.Request) (ms int64, simulated bool) {
	if v := r.Header.Get(Header); v != "" {
		if ms, err := strconv.ParseInt(v, 10, 64); err == nil {
			return ms, true
		}
	}
	return time.Now().UnixMilli(), false
}
