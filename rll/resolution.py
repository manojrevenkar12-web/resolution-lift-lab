"""Question 2: how many of the billed resolutions were real?

The billed number counts "assumed" resolutions: the customer went quiet for a day. Some of those
customers gave up. The truth is known only for a small weekly audit. Estimators of the true
resolution rate per workspace and 4-week block:

- billing_proxy: the billed rate (confirmed + assumed), the status quo
- return_adjusted: billed, minus assumed resolutions whose customer came back within 7 days
- audit_only: the audit sample's mean; unbiased, but ~70 labels per block
- model_only: a classifier trained on audits, averaged over every conversation; precise, can be biased
- ppi: prediction-powered inference (Angelopoulos et al., 2023): the model's mean over all
  conversations, corrected by its average error on the audited ones. Unbiased like the audit,
  tighter when the model is good.

The model is cross-fitted by workspace: a workspace's own audits never train the predictions
used for it, so the correction term is honest.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier

ESTIMATORS = ("billing_proxy", "return_adjusted", "audit_only", "model_only", "ppi")

OUTCOMES = ("confirmed", "assumed", "escalated")
SEGMENTS = ("self_serve", "standard", "enterprise")


def features(conv: pd.DataFrame) -> pd.DataFrame:
    """Observable signals about whether the answer worked. Nothing here is the truth."""
    x = pd.DataFrame(index=conv.index)
    for o in OUTCOMES:
        x[f"outcome_{o}"] = (conv["outcome"] == o).astype(float)
    for s in SEGMENTS:
        x[f"segment_{s}"] = (conv["segment"] == s).astype(float)
    x["log_quiet_seconds"] = np.log1p(conv["quiet_seconds"].astype(float))
    x["returned_7d"] = conv["returned_7d"].astype("float")  # NaN in the last week: not known yet
    x["answer_confidence"] = conv["answer_confidence"].astype(float)
    x["customer_turns"] = conv["customer_turns"].astype(float)
    x["text_length"] = conv["text"].str.len().astype(float)
    x["german"] = (conv["lang"] == "de").astype(float)
    x["weekend"] = (conv["weekday"] >= 5).astype(float)
    return x


def cross_fit(conv: pd.DataFrame, audit: pd.DataFrame, cfg: dict, seed: int) -> np.ndarray:
    """P(resolved) for every conversation, each from a model that never saw its workspace's audits."""
    x = features(conv)
    labelled = conv[["conv_id", "workspace"]].reset_index().merge(audit, on="conv_id")
    spaces = np.array(sorted(conv["workspace"].unique()))
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(spaces), cfg["folds_by_workspace"])
    pred = np.full(len(conv), np.nan)
    pos = pd.Series(np.arange(len(conv)), index=conv.index)
    for fold in folds:
        train = labelled[~labelled["workspace"].isin(fold)]
        model = HistGradientBoostingClassifier(random_state=seed, **cfg["model"])
        model.fit(x.loc[train["index"]], train["true_resolved"].astype(int))
        target = conv.index[conv["workspace"].isin(fold)]
        pred[pos[target].to_numpy()] = model.predict_proba(x.loc[target])[:, 1]
    return pred


def estimate(conv: pd.DataFrame, audit: pd.DataFrame, pred: np.ndarray, cfg: dict) -> pd.DataFrame:
    """One row per workspace x block and estimator, with a 95% interval where the estimator has one."""
    z = stats.norm.ppf(1 - cfg["alpha"] / 2)
    frame = conv[["conv_id", "workspace", "day", "outcome", "returned_7d"]].copy()
    frame["pred"] = pred
    frame["block"] = frame["day"] // cfg["unit_days"]
    frame = frame.merge(audit, on="conv_id", how="left")
    rows = []
    for (w, b), g in frame.groupby(["workspace", "block"]):
        billed = g["outcome"] != "escalated"
        came_back = (g["outcome"] == "assumed") & (g["returned_7d"].fillna(0.0) > 0)
        a = g[g["true_resolved"].notna()]
        y = a["true_resolved"].astype(float).to_numpy()
        n, n_all = len(a), len(g)
        out = {
            "billing_proxy": (billed.mean(), None),
            "return_adjusted": ((billed & ~came_back).mean(), None),
            "audit_only": (y.mean(), z * y.std(ddof=1) / np.sqrt(n)) if n > 1 else (np.nan, None),
            "model_only": (g["pred"].mean(), None),
        }
        if n > 1:
            rect = y - a["pred"].to_numpy()
            est = g["pred"].mean() + rect.mean()
            se = np.sqrt(rect.var(ddof=1) / n + g["pred"].var(ddof=1) / n_all)
            out["ppi"] = (est, z * se)
        else:
            out["ppi"] = (np.nan, None)
        for name, (value, half) in out.items():
            rows.append(
                {
                    "workspace": int(cast(int, w)),
                    "block": int(cast(int, b)),
                    "estimator": name,
                    "estimate": float(value),
                    "low": None if half is None else float(value - half),
                    "high": None if half is None else float(value + half),
                    "conversations": n_all,
                    "audited": n,
                }
            )
    return pd.DataFrame(rows)
