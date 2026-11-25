import json

import numpy as np
import pandas as pd
import pytest

from adtrain import deepui


def test_tsne_keeps_clusters_apart_and_is_repeatable():
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 1, (3, 8)) * 5
    X = np.vstack([c + rng.normal(0, 0.3, (15, 8)) for c in centers])
    a, b = deepui.tsne(X, iters=300), deepui.tsne(X, iters=300)
    assert np.allclose(a, b)
    lab = np.repeat(np.arange(3), 15)
    # each point's nearest point on the map is in its own cluster
    D = ((a[:, None] - a[None]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    assert (lab[D.argmin(1)] == lab).mean() > 0.95


def test_status_lists_datasets_reports_and_hourly_rows(tmp_path):
    (tmp_path / "snapshot").mkdir()
    (tmp_path / "snapshot" / "manifest.json").write_text(json.dumps({"days": [0, 13], "rows": {"impressions": 10}}))
    (tmp_path / "retrain").mkdir()
    (tmp_path / "retrain" / "report.json").write_text(json.dumps({"shift_day": 14}))
    pd.DataFrame({"hour": [0.0], "group": ["hypercasual"], "impressions": [5], "true_credited": [2.0],
                  "daily": [3.0]}).to_csv(tmp_path / "retrain" / "calibration_by_hour.csv", index=False)
    s = deepui.status(tmp_path)
    assert list(s["datasets"]) == ["snapshot"] and s["datasets"]["snapshot"]["days"] == [0, 13]
    h = s["reports"]["retrain"]["hourly"]
    assert h["arms"] == ["daily"] and h["truth"] == "true_credited" and h["rows"][0]["daily"] == 3.0


def test_jobs_refuse_a_missing_dataset(tmp_path):
    with pytest.raises(ValueError, match="no dataset"):
        deepui.Jobs(tmp_path).start("deep")
    with pytest.raises(ValueError, match="unknown job"):
        deepui.Jobs(tmp_path).start("rm -rf")
