// Command bidder runs the reference DSP's bidder and its admin console.
//
//	go run ./cmd/bidder -mode bid -ledger out/ledger
//
// Exchanges, or the simulator, POST bid requests to the bid port at
// /openrtb/{exchange}, and send win, loss, and billing notices to
// /notice/{type}. The admin console and API are on the admin port.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/bgweber/adtechbook/code/dsp/internal/bidder"
	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/console"
	"github.com/bgweber/adtechbook/code/dsp/internal/features"
	"github.com/bgweber/adtechbook/code/dsp/internal/ledger"
	"github.com/bgweber/adtechbook/code/dsp/internal/model"
	"github.com/bgweber/adtechbook/code/dsp/internal/normalize"
	"github.com/bgweber/adtechbook/code/dsp/internal/ops"
	"github.com/bgweber/adtechbook/code/dsp/internal/policy"
	"github.com/bgweber/adtechbook/code/dsp/internal/profile"
	"github.com/bgweber/adtechbook/code/dsp/internal/stats"
)

const version = "0.5.0"

func main() {
	bidAddr := flag.String("bid-addr", ":8080", "bid port, for exchanges")
	adminAddr := flag.String("admin-addr", ":8081", "admin port, for the console")
	mode := flag.String("mode", bidder.ModeBid, "listen: log only, never bid; bid: bid on -bid-share of requests")
	bidShare := flag.Float64("bid-share", 1.0, "share of requests to bid on in bid mode, such as 0.01")
	stage := flag.String("stage", "crawl", "crawl, walk, or run")
	ledgerDir := flag.String("ledger", "out/ledger", "directory for the bid ledger and notice log")
	noticeURL := flag.String("notice-url", "http://localhost:8080",
		"base URL exchanges use to reach this bidder's notice endpoints")
	modelsDir := flag.String("models", "out/models", "directory of published click models")
	trainerURL := flag.String("trainer-url", "",
		"training service the console starts runs on, such as http://localhost:8090")
	opsDir := flag.String("ops", "out/ops", "directory for the kill switches and their audit log")
	throttle := flag.Bool("throttle", true, "spread each campaign's spend over the day (Chapter 12)")
	skew := flag.String("skew", "", "inject a training-serving skew for the Chapter 8 exercise: "+
		"bundle (keep the junk some exchanges append to bundles) or conn (log finer cellular connection types)")
	mmpToken := flag.String("mmp-token", "adsim-mmp",
		"shared secret the MMP sends in X-MMP-Token with each postback; empty accepts any")
	flag.Parse()

	bids, err := ledger.New(*ledgerDir, "bids")
	if err != nil {
		log.Fatal(err)
	}
	notices, err := ledger.New(*ledgerDir, "notices")
	if err != nil {
		log.Fatal(err)
	}
	clicks, err := ledger.New(*ledgerDir, "clicks")
	if err != nil {
		log.Fatal(err)
	}
	raw, err := ledger.New(*ledgerDir, "raw")
	if err != nil {
		log.Fatal(err)
	}
	postbacks, err := ledger.New(*ledgerDir, "postbacks")
	if err != nil {
		log.Fatal(err)
	}
	models := &model.Registry{Dir: *modelsDir}
	if v, err := models.LoadDeployed(); err != nil {
		log.Printf("deployed model %s not loaded, bidding with priors: %v", v, err)
	} else if v != "" {
		log.Printf("deployed click model: %s", v)
	}
	st, book, profiles := stats.New(), campaigns.NewBook(), profile.NewStore()
	opsys := ops.New(*opsDir)
	opsys.Throttle = *throttle
	for _, k := range strings.Split(*skew, ",") {
		switch strings.TrimSpace(k) {
		case "":
		case "bundle":
			normalize.SkewKeepBundleSuffix = true
		case "conn":
			features.SkewFineConnection = true
		default:
			log.Fatalf("unknown -skew %q: use bundle or conn", k)
		}
		if k != "" {
			log.Printf("skew injected: %s (Chapter 8 exercise)", k)
			opsys.Note("training-serving skew injected in the bidder: "+k, "startup", 0)
		}
	}
	b := &bidder.Bidder{Mode: *mode, Version: version, BidShare: *bidShare, NoticeURL: *noticeURL,
		Policy: policy.DefaultCrawl(), Book: book, Profiles: profiles,
		Bids: bids, Notices: notices, Clicks: clicks, Postbacks: postbacks, Raw: raw, MMPToken: *mmpToken,
		Stats: st, Models: models, Ops: opsys}

	bidMux := http.NewServeMux()
	bidMux.Handle("/openrtb/", b)
	bidMux.Handle("/notice/", b)
	bidMux.Handle("/click", b)
	bidMux.Handle("/postback", b)
	bidSrv := &http.Server{Addr: *bidAddr, Handler: bidMux,
		ReadTimeout: time.Second, WriteTimeout: time.Second}
	adminSrv := &http.Server{Addr: *adminAddr, Handler: console.Handler(console.Deps{
		Info:  console.Info{Version: version, Mode: *mode, Stage: *stage, BidShare: *bidShare},
		Stats: st, Book: book, Profiles: profiles, Bids: bids, Notices: notices, Clicks: clicks,
		Postbacks: postbacks, Models: models, Ops: opsys, Trainer: &console.Trainer{URL: *trainerURL, Models: models}})}

	go serve(bidSrv, "bid")
	go serve(adminSrv, "admin")
	log.Printf("bidder %s: mode=%s bid-share=%g stage=%s bid=%s console=http://localhost%s",
		version, *mode, *bidShare, *stage, *bidAddr, *adminAddr)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	bidSrv.Shutdown(ctx)
	adminSrv.Shutdown(ctx)
	bids.Close()
	notices.Close()
	clicks.Close()
	postbacks.Close()
	raw.Close()
	log.Printf("ledger: %d bid records (%d dropped), %d notices (%d dropped), %d clicks",
		bids.Written(), bids.Dropped(), notices.Written(), notices.Dropped(), clicks.Written())
}

func serve(s *http.Server, name string) {
	if err := s.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("%s server: %v", name, err)
	}
}
