from adtrain import datasets, gbm, pipeline

from test_gbm import synthetic


def checks(lst):
    return {c["name"]: c["passed"] for c in lst}


def test_model_checks_stop_a_broken_model(tmp_path):
    df, m = synthetic()
    d = datasets.write(df, m, tmp_path / "datasets")
    good, rep = gbm.train(d, tmp_path / "models", log=lambda *_: None)
    c = checks(pipeline.validate_bundle(good, rep, df))
    assert c["predictions vary"] and c["dataset passed"] and c["vs. champion"]

    # every feature missing: the models learn the base rate and nothing else
    bad, brep = gbm.train(d, tmp_path / "models", inject="no-features", log=lambda *_: None)
    c = checks(pipeline.validate_bundle(bad, brep, df, champion_dir=good))
    assert not c["predictions vary"] and not c["click AUC"]
    assert not c["click vs. champion"]

    # the same data and code make the same model, so it ties the champion
    again, arep = gbm.train(d, tmp_path / "models", log=lambda *_: None)
    c = checks(pipeline.validate_bundle(again, arep, df, champion_dir=good))
    assert c["click vs. champion"] and c["install_click vs. champion"]


def test_aliases(tmp_path):
    assert pipeline.read_alias(tmp_path, "CHALLENGER") == ""
    pipeline.write_alias(tmp_path, "CHALLENGER", "gbm-0002")
    assert pipeline.read_alias(tmp_path, "CHALLENGER") == "gbm-0002"
