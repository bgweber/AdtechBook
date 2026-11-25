// Command deepcheck loads the deep model bundle adtrain exports (Chapter
// 28), checks it against PyTorch's golden vectors, benchmarks it per request
// with growing numbers of candidates, and checks the versioned app vectors
// and device states the bidder scores with dot products:
//
//	python -m adtrain export --snapshot out/snapshot --out out/export
//	deepcheck -dir out/export
//
// In Docker: docker compose run --rm --no-deps --entrypoint deepcheck bidder -dir /out/export
// The console's Deep Models tab runs the same check in the bidder.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	"github.com/bgweber/adtechbook/code/dsp/internal/deep"
)

func main() {
	dir := flag.String("dir", "/out/export", "the folder adtrain export wrote")
	reps := flag.Int("n", 2000, "requests timed per configuration")
	flag.Parse()
	runtime.GOMAXPROCS(1) // one core, as one bidder worker sees it
	if _, err := deep.Check(*dir, *reps, os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, "deepcheck:", err)
		fmt.Fprintln(os.Stderr, "Run first: python -m adtrain export --snapshot /out/snapshot --out /out/export")
		os.Exit(1)
	}
}
