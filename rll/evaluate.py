"""Runs every method on a world's observed data and grades it against the world's truth.

Methods receive `Observed` only; the truth is joined in here, after they have answered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from rll import lift, resolution, topics
from rll.world.sim import World


def grade_topics(world: World, cfg: dict) -> pd.DataFrame:
    obs = world.observed().conversations
    truth = world.conversations.set_index("conv_id")["true_topic"]
    sample = obs[obs["day"] < cfg["sample_days"]]
    rows = []
    for w, g in sample.groupby("workspace"):
        g = g.reset_index(drop=True)
        true = truth.loc[g["conv_id"]].to_numpy()
        gap = world.gaps[int(w)]
        for method in topics.METHODS:
            r = topics.run(g, method, cfg, seed=world.seed)
            majority = pd.Series(true).groupby(r.labels).agg(lambda s: s.value_counts().index[0])
            flagged = [majority.get(c) for c in r.gap_clusters]
            rows.append(
                {
                    "workspace": int(w),
                    "method": method,
                    "ari": adjusted_rand_score(true, r.labels),
                    "nmi": normalized_mutual_info_score(true, r.labels),
                    "assigned": float((r.labels >= 0).mean()),
                    "clusters": int(len(set(r.labels) - {-1})),
                    "gap_found": gap in flagged,
                    "false_gap_flags": int(sum(f != gap for f in flagged)),
                    "lang": g["lang"].iloc[0],
                }
            )
    return pd.DataFrame(rows)


def grade_resolution(world: World, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    obs = world.observed()
    conv = obs.conversations
    pred = resolution.cross_fit(conv, obs.audit_labels, cfg, seed=world.seed)
    est = resolution.estimate(conv, obs.audit_labels, pred, cfg)
    truth = world.truth()
    truth["block"] = truth["day"] // cfg["unit_days"]
    target = truth.groupby(["workspace", "block"]).agg(true_rate=("true_resolved", "mean"), billed_not_resolved=("billable", lambda b: 0.0))
    t2 = truth.assign(bnr=truth["billable"] & ~truth["true_resolved"].astype(bool))
    target["billed_not_resolved"] = t2.groupby(["workspace", "block"])["bnr"].sum()
    target["billed"] = truth.groupby(["workspace", "block"])["billable"].sum()
    est = est.merge(target.reset_index(), on=["workspace", "block"])
    est["error"] = est["estimate"] - est["true_rate"]
    est["covered"] = np.where(est["low"].notna(), (est["low"] <= est["true_rate"]) & (est["true_rate"] <= est["high"]), np.nan)
    est["width"] = est["high"] - est["low"]
    return est, pred


def grade_lift(world: World, rcfg: dict, lcfg: dict) -> tuple[pd.DataFrame, dict]:
    fixes = [f.__dict__ for f in world.fixes]
    truth = world.true_fix_effects
    out = []
    extra: dict = {}
    views: dict[str, pd.DataFrame] = {}
    for view in ("full_rollout", "ab_holdout"):
        obs = world.observed(view)
        conv = obs.conversations.copy()
        conv["pred"] = resolution.cross_fit(conv, obs.audit_labels, rcfg, seed=world.seed)
        conv["billed"] = (conv["outcome"] != "escalated").astype(float)
        views[view] = conv
    for design in lift.DESIGNS:
        conv = views["ab_holdout" if design == "ab_holdout" else "full_rollout"]
        est = lift.estimate_all(conv, fixes, design, lcfg)
        out.append(est)
    per_fix = pd.concat(out, ignore_index=True).merge(truth, on="fix_id")
    # how small an effect could one article's A/B detect?
    ab = views["ab_holdout"]
    mdes = []
    for f in fixes:
        post = ab[
            (ab["workspace"] == f["workspace"])
            & (ab["cited_article"] == f["article"])
            & ab["day"].between(f["day"], f["day"] + lcfg["post_days"] - 1)
            & ab["arm"].notna()
        ]
        if len(post) >= lcfg["min_conversations"]:
            mdes.append(
                lift.minimum_detectable_effect(
                    post["pred"].mean(), len(post) / 2, lcfg["alpha"], lcfg["power"], sd=float(post["pred"].std(ddof=1))
                )
            )
    extra["mde_median"] = float(np.median(mdes)) if mdes else np.nan
    return per_fix, extra


def summarize_lift(per_fix: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for design, g in per_fix.groupby("design"):
        ok = g.dropna(subset=["estimate"])
        err = ok["estimate"] - ok["true_effect"]
        claimed = ok[ok["low"] > 0]
        pooled = lift.pooled_effect(ok, alpha)
        rows.append(
            {
                "design": design,
                "fixes": int(len(g)),
                "estimated": int(len(ok)),
                "bias": float(err.mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "coverage": float(((ok["low"] <= ok["true_effect"]) & (ok["true_effect"] <= ok["high"])).mean()),
                "claimed_wins": int(len(claimed)),
                "false_wins": int((claimed["true_effect"] <= 0).sum()),
                "pooled": pooled["estimate"],
                "pooled_low": pooled["low"],
                "pooled_high": pooled["high"],
                "true_mean": float(ok["true_effect"].mean()),
            }
        )
    return pd.DataFrame(rows)
