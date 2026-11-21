import numpy as np
import pandas as pd

from rll import resolution
from rll.config import load_methods

CFG = load_methods()["resolution"]


def test_ppi_corrects_a_biased_model():
    rng = np.random.default_rng(0)
    n = 4000
    truth = rng.random(n) < 0.7
    pred = np.clip(truth * 0.8 + 0.15 + rng.normal(0, 0.1, n), 0, 1)  # biased upwards for failures
    conv = pd.DataFrame({"conv_id": np.arange(n), "workspace": 0, "day": 0, "outcome": "assumed", "returned_7d": 0.0})
    audited = rng.choice(n, 300, replace=False)
    audit = pd.DataFrame({"conv_id": audited, "true_resolved": truth[audited]})
    est = resolution.estimate(conv, audit, pred, CFG).set_index("estimator")
    assert abs(est.loc["ppi", "estimate"] - truth.mean()) < 0.02
    assert abs(est.loc["model_only", "estimate"] - truth.mean()) > abs(est.loc["ppi", "estimate"] - truth.mean())
    assert est.loc["ppi", "low"] < truth.mean() < est.loc["ppi", "high"]


def test_cross_fit_never_trains_on_a_workspace_own_audits(small_world, monkeypatch):
    obs = small_world.observed()
    seen: list[set[int]] = []
    real = resolution.HistGradientBoostingClassifier

    class Spy(real):
        def fit(self, x, y):
            seen.append(set(obs.conversations.loc[x.index, "workspace"]))
            return super().fit(x, y)

    monkeypatch.setattr(resolution, "HistGradientBoostingClassifier", Spy)
    cfg = {**CFG, "folds_by_workspace": 3}
    pred = resolution.cross_fit(obs.conversations, obs.audit_labels, cfg, seed=0)
    assert np.isfinite(pred).all() and len(seen) == 3
    assert all(len(s) == 2 for s in seen)  # each model saw the other two workspaces only


def test_features_hold_no_truth(small_world):
    x = resolution.features(small_world.observed().conversations)
    assert not any("true" in c or "resolved" in c for c in x.columns)
