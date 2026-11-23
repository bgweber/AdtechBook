package experiment

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/bgweber/adtechbook/code/dsp/internal/campaigns"
	"github.com/bgweber/adtechbook/code/dsp/internal/gbm"
	"github.com/bgweber/adtechbook/code/dsp/internal/policy"
)

func setup(t *testing.T) *Manager {
	t.Helper()
	root := t.TempDir()
	src := filepath.Join("..", "gbm", "testdata", "gbm-fixture")
	for _, v := range []string{"gbm-0001", "gbm-0002"} {
		dst := filepath.Join(root, gbm.Family, v)
		os.MkdirAll(dst, 0o755)
		ents, _ := os.ReadDir(src)
		for _, e := range ents {
			b, _ := os.ReadFile(filepath.Join(src, e.Name()))
			os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644)
		}
	}
	reg := &gbm.Registry{Dir: root}
	if res := reg.Deploy("gbm-0001", false); !res.OK {
		t.Fatal(res.Message)
	}
	return &Manager{Dir: filepath.Join(root, "experiments"), Registry: reg, Book: campaigns.NewBook()}
}

func TestArmsAreStableAndSized(t *testing.T) {
	l := &Live{Kind: Canary, Share: 0.25, Salt: "exp-0001"}
	n := 0
	for i := 0; i < 20000; i++ {
		ifa := string(rune('a'+i%26)) + string(rune(i))
		a := l.Arm(ifa)
		if a != l.Arm(ifa) {
			t.Fatal("a device changed arms")
		}
		if a == campaigns.ArmTreatment {
			n++
		}
	}
	if n < 4500 || n > 5500 {
		t.Fatalf("treatment got %d of 20000 devices for a 25%% share", n)
	}
	// growing the share keeps every treatment device in treatment
	l2 := &Live{Kind: Canary, Share: 0.5, Salt: "exp-0001"}
	for i := 0; i < 2000; i++ {
		ifa := string(rune(i))
		if l.Arm(ifa) == campaigns.ArmTreatment && l2.Arm(ifa) != campaigns.ArmTreatment {
			t.Fatal("a device left the canary when its share grew")
		}
	}
}

// feed sends decisions to a canary: control and treatment, with the
// canary's predictions scaled by k against the champion's.
func feed(m *Manager, n int, k float64, from, step int64) int64 {
	ts := from
	for i := 0; i < n; i++ {
		l := m.Live()
		if l == nil {
			return ts
		}
		for _, arm := range []int{campaigns.ArmControl, campaigns.ArmTreatment} {
			d := policy.Decision{Action: "bid", CampaignID: 1, PClick: 0.02, PClickFrom: "model",
				PInstallClick: 0.1, InstallRateFrom: "model"}
			var champ *policy.Prediction
			if arm == campaigns.ArmTreatment {
				champ = &policy.Prediction{PClick: 0.02, PInstallClick: 0.1}
				d.PClick *= k
				d.PInstallClick *= k
			}
			m.Observe(l, arm, "interstitial", &d, champ, ts)
		}
		ts += step
	}
	return ts
}

func TestCanaryPromotesOrRollsBack(t *testing.T) {
	m := setup(t)
	if _, err := m.Start(Config{Kind: Canary, Challenger: "gbm-0002", StageHours: 1}, 1); err != nil {
		t.Fatal(err)
	}
	if _, err := m.Start(Config{Kind: AB, Challenger: "gbm-0002"}, 1); err == nil {
		t.Fatal("started a second experiment")
	}
	// a healthy canary passes three one-hour stages and is promoted
	feed(m, 2000, 1.05, 1, 10_000)
	e := m.Current()
	if e.Status != "promoted" || m.Registry.Current().Spec.Version != "gbm-0002" {
		t.Fatalf("status %s, events %+v", e.Status, e.Events)
	}

	// a broken one is rolled back at the first check with enough data
	m2 := setup(t)
	m2.Start(Config{Kind: Canary, Challenger: "gbm-0002", StageHours: 1}, 1)
	feed(m2, 2000, 3, 1, 10_000)
	e = m2.Current()
	if e.Status != "rolled back" || m2.Live() != nil || m2.Registry.Current().Spec.Version != "gbm-0001" {
		t.Fatalf("status %s: %s", e.Status, e.Reason)
	}
}

func TestBudgetSplit(t *testing.T) {
	b := campaigns.NewBook()
	b.Load([]campaigns.Campaign{{ID: 1, Goal: "CPI", Geos: "USA", DailyBudget: 10, Target: 2}},
		[]campaigns.Creative{{ID: 1, CampaignID: 1, Format: "static"}})
	b.SetSplit(0.2) // treatment gets $2 of the $10
	b.AddSpendArm(0, 1, campaigns.ArmTreatment, 2)
	if c, _ := b.EligibleArm("USA", "banner", 0, campaigns.ArmTreatment); len(c) != 0 {
		t.Fatal("the treatment arm spent its part and can still bid")
	}
	if c, _ := b.EligibleArm("USA", "banner", 0, campaigns.ArmControl); len(c) != 1 || c[0].Remaining != 1 {
		t.Fatalf("the control arm lost budget it didn't spend: %+v", c)
	}
	b.SetSplit(0)
	if c, _ := b.EligibleArm("USA", "banner", 0, campaigns.ArmTreatment); len(c) != 1 {
		t.Fatal("after the test, the campaign's whole budget is shared again")
	}
}
