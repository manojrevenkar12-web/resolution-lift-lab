"""Question 1: what are customers writing about, and which topics have no article at all?

Methods group the first `sample_days` of a workspace's conversations into topics from the text
only, then flag clusters whose conversations are answered from many unrelated articles: the agent
is improvising, because there is no article for that question (a content gap).

- keyword_rules: the status quo, hand-kept English phrases per topic (German text goes unmatched)
- by_citation: one topic per article the agent cited (cannot see a question no article covers)
- word_kmeans: word TF-IDF, k-means, k chosen by silhouette
- char_kmeans: character n-gram TF-IDF (robust to typos and German compounds), SVD, k-means
- char_kmeans_cited: the same text clusters, deliberately too many, then merged where the agent's
  citations agree (one article dominates both) and, among clusters no article dominates, where the
  texts are close. Citations are weak labels the product already logs; gaps are what they cannot explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

METHODS = ("keyword_rules", "by_citation", "word_kmeans", "char_kmeans", "char_kmeans_cited")


@dataclass
class TopicResult:
    labels: np.ndarray  # cluster per conversation; -1 = unassigned
    gap_clusters: list[int]  # clusters flagged as content gaps


def keyword_rules(texts: pd.Series, rules: dict[str, list[str]]) -> np.ndarray:
    names = list(rules)
    out = np.full(len(texts), -1)
    lower = texts.str.lower().to_numpy()
    for i, t in enumerate(lower):
        for k, name in enumerate(names):
            if any(p in t for p in rules[name]):
                out[i] = k
                break
    return out


def _embed(texts: pd.Series, kind: str, components: int, seed: int) -> np.ndarray:
    if kind == "word":
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    else:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    x = vec.fit_transform(texts)
    k = min(components, x.shape[1] - 1, x.shape[0] - 1)
    z = TruncatedSVD(n_components=k, random_state=seed).fit_transform(x)
    return normalize(z)


def _kmeans_best(z: np.ndarray, k_range: tuple[int, int], seed: int) -> np.ndarray:
    best, best_score = None, -1.0
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(z), size=min(len(z), 1500), replace=False)
    for k in range(k_range[0], k_range[1] + 1):
        labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(z)
        score = silhouette_score(z[sample], labels[sample])
        if score > best_score:
            best, best_score = labels, score
    assert best is not None
    return best


def _merge_by_citation(z: np.ndarray, labels: np.ndarray, cited: np.ndarray, cfg: dict) -> np.ndarray:
    frame = pd.DataFrame({"c": labels, "a": cited})
    top = frame.groupby("c")["a"].agg(lambda s: s.value_counts().index[0])
    share = frame.groupby("c")["a"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    merged: dict[int, int] = {}
    owner: dict[str, int] = {}
    loose = [c for c in share.index if share[c] < cfg["merge_citation_share"]]
    for c in share.index:
        if share[c] >= cfg["merge_citation_share"]:
            merged[c] = owner.setdefault(top[c], c)
    # clusters no article explains: join those whose texts are close
    cents = {c: normalize(z[labels == c].mean(axis=0, keepdims=True))[0] for c in loose}
    for i, a in enumerate(loose):
        merged.setdefault(a, a)
        for b in loose[i + 1 :]:
            if b not in merged and float(cents[a] @ cents[b]) >= cfg["merge_text_similarity"]:
                merged[b] = merged[a]
    return np.array([merged.get(lab, lab) for lab in labels])


def discover(texts: pd.Series, method: str, cfg: dict, seed: int, cited: pd.Series | None = None) -> np.ndarray:
    if method == "keyword_rules":
        return keyword_rules(texts, cfg["keyword_rules"])
    if method == "by_citation":
        assert cited is not None
        return pd.factorize(cited)[0]
    kind = "word" if method == "word_kmeans" else "char"
    if method == "char_kmeans_cited":
        assert cited is not None
        z = _embed(texts, kind, cfg["svd_components"], seed)
        labels = KMeans(n_clusters=cfg["overcluster_k"], n_init=4, random_state=seed).fit_predict(z)
        return _merge_by_citation(z, labels, cited.to_numpy(), cfg)
    z = _embed(texts, kind, cfg["svd_components"], seed)
    labels = _kmeans_best(z, tuple(cfg["k_range"]), seed)
    # tiny clusters are noise, not topics
    counts = pd.Series(labels).value_counts(normalize=True)
    small = set(counts[counts < cfg["min_cluster_share"]].index)
    return np.array([-1 if lab in small else lab for lab in labels])


def flag_gaps(labels: np.ndarray, cited: pd.Series, cfg: dict) -> list[int]:
    """A cluster is a gap when no article dominates its citations: the agent answers it from
    whatever is closest, because nothing was written for it."""
    frame = pd.DataFrame({"c": labels, "a": cited.to_numpy()})
    frame = frame[frame["c"] >= 0]
    top_share = frame.groupby("c")["a"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    return [int(cast(int, c)) for c, share in top_share.items() if share <= cfg["gap_max_citation_share"]]


def run(conv: pd.DataFrame, method: str, cfg: dict, seed: int) -> TopicResult:
    labels = discover(conv["text"], method, cfg, seed, cited=conv["cited_article"])
    return TopicResult(labels=labels, gap_clusters=flag_gaps(labels, conv["cited_article"], cfg))
