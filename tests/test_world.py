import numpy as np

from rll.world.sim import OBSERVED_COLUMNS, simulate
from tests.conftest import small_config

TRUTH = {"true_topic", "is_gap", "resolved__new", "resolved__old", "p_true__new", "p_true__old"}


def test_same_seed_same_world():
    a, b = simulate(3, small_config()), simulate(3, small_config())
    assert a.conversations.equals(b.conversations)
    assert [f.__dict__ for f in a.fixes] == [f.__dict__ for f in b.fixes]


def test_observed_view_holds_no_truth(small_world):
    obs = small_world.observed()
    assert not TRUTH & set(obs.conversations.columns)
    assert list(obs.conversations.columns) == OBSERVED_COLUMNS
    # audit labels exist only for the audited sample
    assert len(obs.audit_labels) == len(small_world.audits) < len(obs.conversations) / 10


def test_both_designs_share_all_luck_outside_the_holdout(small_world):
    full = small_world.observed("full_rollout").conversations
    ab = small_world.observed("ab_holdout").conversations
    outside = ab["arm"].isna() | ab["arm"].eq("new")
    assert full.loc[outside, "outcome"].equals(ab.loc[outside, "outcome"])
    assert set(ab["arm"].dropna()) == {"old", "new"}


def test_a_content_gap_is_never_cited(small_world):
    c = small_world.conversations
    for w, gap in small_world.gaps.items():
        assert not ((c["workspace"] == w) & (c["cited_article"] == gap)).any()
        assert ((c["workspace"] == w) & (c["true_topic"] == gap)).any()


def test_assumed_resolutions_include_customers_who_gave_up(small_world):
    c = small_world.conversations
    assumed = c[c["outcome__new"] == "assumed"]
    assert 0.05 < 1 - assumed["resolved__new"].mean() < 0.5
    assert (c[c["outcome__new"] == "confirmed"]["resolved__new"]).all()


def test_fixes_mostly_follow_a_bad_week(small_world):
    fixes = small_world.fixes
    assert fixes, "the content team fixed nothing"
    picked = [f.selected_on_rate for f in fixes if f.reason == "lowest_rate"]
    overall = (small_world.conversations["outcome__new"] != "escalated").mean()
    assert np.mean(picked) < overall


def test_true_effects_come_from_the_potential_outcomes(small_world):
    eff = small_world.true_fix_effects
    assert len(eff) == len(small_world.fixes)
    assert eff["true_effect"].between(-0.5, 0.5).all()
