package gbm

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func copyFixture(t *testing.T, root, version string) string {
	t.Helper()
	src := filepath.Join("testdata", "gbm-fixture")
	dst := filepath.Join(root, Family, version)
	if err := os.MkdirAll(dst, 0o755); err != nil {
		t.Fatal(err)
	}
	ents, _ := os.ReadDir(src)
	for _, e := range ents {
		b, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dst, e.Name()), b, 0o644)
	}
	os.WriteFile(filepath.Join(dst, "report.json"), []byte(`{"dataset_passed": true}`), 0o644)
	return dst
}

func TestDeployChecks(t *testing.T) {
	root := t.TempDir()
	r := &Registry{Dir: root}

	// a good bundle deploys, and survives a restart
	copyFixture(t, root, "gbm-0001")
	if res := r.Deploy("gbm-0001", false); !res.OK {
		t.Fatal(res.Message)
	}
	r2 := &Registry{Dir: root}
	if v, err := r2.LoadDeployed(); err != nil || v != "gbm-0001" || r2.Current() == nil {
		t.Fatalf("after restart: %q %v", v, err)
	}

	// trained on a dataset that failed validation: needs force
	d := copyFixture(t, root, "gbm-0002")
	os.WriteFile(filepath.Join(d, "report.json"), []byte(`{"dataset_passed": false}`), 0o644)
	if res := r.Deploy("gbm-0002", false); res.OK || !strings.Contains(res.Message, "force") {
		t.Fatalf("deployed without force: %s", res.Message)
	}
	if res := r.Deploy("gbm-0002", true); !res.OK {
		t.Fatal(res.Message)
	}

	// golden predictions that don't match: refused, even forced
	d = copyFixture(t, root, "gbm-0003")
	g, _ := os.ReadFile(filepath.Join(d, "golden.jsonl"))
	bad := strings.Replace(string(g), `"click": 0.`, `"click": 0.9`, 1)
	os.WriteFile(filepath.Join(d, "golden.jsonl"), []byte(bad), 0o644)
	if res := r.Deploy("gbm-0003", true); res.OK {
		t.Fatal("deployed a bundle whose golden examples don't match")
	}
	if r.Current() == nil {
		t.Fatal("a refused deploy changed the serving bundle")
	}
}

func TestPromoteAndRollback(t *testing.T) {
	root := t.TempDir()
	r := &Registry{Dir: root}
	copyFixture(t, root, "gbm-0001")
	copyFixture(t, root, "gbm-0002")
	if res := r.Promote(); res.OK {
		t.Fatal("promoted with no challenger")
	}
	r.Deploy("gbm-0001", false)
	os.WriteFile(filepath.Join(root, Family, "CHALLENGER"), []byte("gbm-0002\n"), 0o644)
	if l := r.List(); !l[0].Challenger || l[0].Version != "gbm-0002" || !l[1].Deployed {
		t.Fatalf("list: %+v", l)
	}
	if res := r.Promote(); !res.OK {
		t.Fatal(res.Message)
	}
	if r.List()[0].Challenger {
		t.Fatal("the champion is still shown as the challenger")
	}
	if res := r.Rollback(); !res.OK || r.Current() == nil {
		t.Fatal(res.Message)
	}
	h := r.History(0)
	if len(h) != 3 || h[0].Action != "rollback" || h[0].Version != "gbm-0001" ||
		h[0].Previous != "gbm-0002" || h[1].Action != "promote" {
		t.Fatalf("history: %+v", h)
	}

	// a bundle that failed the pipeline's model checks needs force
	d := copyFixture(t, root, "gbm-0003")
	os.WriteFile(filepath.Join(d, "validation.json"),
		[]byte(`{"checks": [{"name": "predictions vary", "passed": false}]}`), 0o644)
	if res := r.Deploy("gbm-0003", false); res.OK || !strings.Contains(res.Message, "predictions vary") {
		t.Fatalf("deployed a bundle that failed its checks: %s", res.Message)
	}
}
