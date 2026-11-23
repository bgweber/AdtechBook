package profile

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestSaveLoad(t *testing.T) {
	a := NewStore()
	a.Observe("d1", "app.a", 1000, 0)
	a.Observe("d1", "app.b", 2000, 0)
	a.RecordImpression("d1", 2000, 0)
	a.RecordConversion("d1", "install")
	a.Observe("d2", "", 3000, 0)
	path := filepath.Join(t.TempDir(), "state", "profiles.gob.gz")
	if n, err := a.Save(path); err != nil || n != 2 {
		t.Fatalf("save: %d %v", n, err)
	}
	b := NewStore()
	if n, err := b.Load(path); err != nil || n != 2 {
		t.Fatalf("load: %d %v", n, err)
	}
	sa, sb := a.Observe("d1", "app.c", 4000, 0), b.Observe("d1", "app.c", 4000, 0)
	if !reflect.DeepEqual(sa, sb) || sb.Installs != 1 || sb.ImpsToday != 1 || sb.Requests != 2 {
		t.Fatalf("profiles differ after a restart:\n%+v\n%+v", sa, sb)
	}
	if n, err := NewStore().Load(filepath.Join(t.TempDir(), "none")); n != 0 || err != nil {
		t.Fatalf("missing file: %d %v", n, err)
	}
}
