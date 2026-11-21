import numpy as np
import pandas as pd

from rll import topics
from rll.config import load_methods

CFG = load_methods()["topics"]


def test_keyword_rules_leave_german_unmatched():
    texts = pd.Series(["i forgot my password", "ich habe mein passwort vergessen"])
    labels = topics.keyword_rules(texts, CFG["keyword_rules"])
    assert labels[0] >= 0 and labels[1] == -1


def test_a_cluster_answered_from_many_articles_is_a_gap():
    labels = np.array([0] * 10 + [1] * 10)
    cited = pd.Series(["a"] * 10 + ["a", "b", "c", "d", "e", "f", "a", "b", "c", "d"])
    assert topics.flag_gaps(labels, cited, CFG) == [1]


def test_citation_merge_finds_the_gap_in_a_real_workspace(small_world):
    obs = small_world.observed().conversations
    g = obs[(obs["workspace"] == 0) & (obs["day"] < 14)].reset_index(drop=True)
    r = topics.run(g, "char_kmeans_cited", CFG, seed=1)
    truth = small_world.conversations.set_index("conv_id").loc[g["conv_id"], "true_topic"].to_numpy()
    flagged = [pd.Series(truth[r.labels == c]).value_counts().index[0] for c in r.gap_clusters]
    assert small_world.gaps[0] in flagged


def test_by_citation_can_never_see_a_gap(small_world):
    obs = small_world.observed().conversations
    g = obs[obs["workspace"] == 1].reset_index(drop=True)
    r = topics.run(g, "by_citation", CFG, seed=1)
    assert r.gap_clusters == []
