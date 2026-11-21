import json

import pytest

from rll import report, run


def test_heldout_runs_once(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REPORTS", tmp_path)
    monkeypatch.setattr(run, "LOCK", tmp_path / "heldout.lock")
    (tmp_path / "heldout.lock").write_text(json.dumps({"ran_at": "x"}))
    with pytest.raises(SystemExit, match="run once"):
        run.run("heldout")


def test_gate_flags_a_pooled_estimate_that_misses_the_truth():
    doc = json.loads((run.ROOT / "reports" / "dev.json").read_text())
    gates = {
        "topics": {"ari_min": 0, "gaps_found_min": 0, "false_flags_per_workspace_max": 99},
        "resolution": {"ppi_coverage_min": 0, "ppi_mae_max": 1},
        "lift": {"pooled_error_max": 0.01},
    }
    assert report.gate(doc, gates) == []
    doc["lift"]["ab_holdout"]["pooled"] += 0.05
    assert any("pooled" in f for f in report.gate(doc, gates))
