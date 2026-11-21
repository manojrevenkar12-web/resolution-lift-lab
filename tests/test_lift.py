import numpy as np
import pandas as pd

from rll import lift
from rll.config import load_methods

CFG = load_methods()["lift"]


def frame(effect: float, dip: float, seed: int = 0) -> pd.DataFrame:
    """One article fixed on day 21 after a bad week (dip), with an A/B holdout."""
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(0, 42):
        for art in ("fixed", "other1", "other2"):
            for _ in range(40):
                p = 0.7
                if art == "fixed" and 14 <= day < 21:
                    p -= dip
                arm = None
                if art == "fixed" and 21 <= day < 35:
                    arm = "new" if rng.random() < 0.5 else "old"
                    p += effect if arm == "new" else 0
                elif art == "fixed" and day >= 35:
                    p += effect
                rows.append({"workspace": 0, "day": day, "cited_article": art, "arm": arm, "pred": p, "billed": float(rng.random() < p)})
    return pd.DataFrame(rows)


FIX = {"fix_id": 0, "workspace": 0, "article": "fixed", "day": 21}


def test_before_after_mistakes_a_bad_week_for_an_effect():
    conv = frame(effect=0.0, dip=0.1)
    naive = lift.estimate_fix(conv, FIX, "before_after_model", CFG, set())
    ab = lift.estimate_fix(conv, FIX, "ab_holdout", CFG, set())
    rtm = lift.estimate_fix(conv, FIX, "did_rtm", CFG, set())
    assert naive["estimate"] > 0.09
    assert abs(ab["estimate"]) < 1e-9 and abs(rtm["estimate"]) < 1e-9


def test_ab_recovers_a_real_effect():
    conv = frame(effect=0.05, dip=0.1)
    assert abs(lift.estimate_fix(conv, FIX, "ab_holdout", CFG, set())["estimate"] - 0.05) < 1e-9


def test_pooled_effect_weights_by_precision():
    per = pd.DataFrame({"estimate": [0.0, 0.1], "se": [0.01, 0.1]})
    out = lift.pooled_effect(per, 0.05)
    assert 0 < out["estimate"] < 0.01 and out["fixes"] == 2


def test_minimum_detectable_effect_shrinks_with_volume():
    small, large = (lift.minimum_detectable_effect(0.7, n, 0.05, 0.8) for n in (50, 5000))
    assert small > 0.2 > 0.03 > large
