"""Question 3: did a content fix raise resolutions?

The content team fixes, mostly, the article that looked worst last week. That week was partly bad
luck, so the next weeks look better anyway (regression to the mean); the agent's model also improved
on one day for everyone. Designs, each giving an estimate and a 95% interval per fix:

- before_after_billed: billed rate in the 14 days after vs the 7 days before (what a dashboard shows)
- before_after_model: the same, on the model's estimate of true resolution
- did: difference-in-differences against the workspace's other articles, same windows
- did_rtm: the same, but the "before" window skips the week that triggered the fix
- ab_holdout: half the conversations keep the old article for 14 days; new minus old

`pooled_effect` combines per-fix estimates into the average effect of a fix, which is the question
the product actually needs answered ("do Recommendations' fixes work?").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

DESIGNS = ("before_after_billed", "before_after_model", "did", "did_rtm", "ab_holdout")


def _mean_var(v: pd.Series) -> tuple[float, float, int]:
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    return float(v.mean()), float(v.var(ddof=1)) if n > 1 else np.nan, n


def _window(conv: pd.DataFrame, w: int, start: int, end: int) -> pd.DataFrame:
    return conv[(conv["workspace"] == w) & (conv["day"] >= start) & (conv["day"] < end)]


def estimate_fix(conv: pd.DataFrame, fix: dict, design: str, cfg: dict, fixed_nearby: set[str]) -> dict:
    """conv has `billed` and `pred` columns; returns estimate, se for one fix."""
    d, w, art = fix["day"], fix["workspace"], fix["article"]
    post_end = d + cfg["post_days"]
    if design == "ab_holdout":
        post = _window(conv, w, d, post_end)
        post = post[post["cited_article"] == art]
        new, old = post[post["arm"] == "new"]["pred"], post[post["arm"] == "old"]["pred"]
        m1, v1, n1 = _mean_var(new)
        m0, v0, n0 = _mean_var(old)
        if min(n0, n1) < cfg["min_conversations"] // 2:
            return {"estimate": np.nan, "se": np.nan}
        return {"estimate": m1 - m0, "se": np.sqrt(v1 / n1 + v0 / n0)}

    outcome = "billed" if design == "before_after_billed" else "pred"
    if design == "did_rtm":
        pre_start, pre_end = d - cfg["rtm_pre"][0], d - cfg["rtm_pre"][1]
    else:
        pre_start, pre_end = d - cfg["pre_days"], d
    if pre_start < 0:
        return {"estimate": np.nan, "se": np.nan}
    pre_all, post_all = _window(conv, w, pre_start, pre_end), _window(conv, w, d, post_end)
    pre, post = pre_all[pre_all["cited_article"] == art], post_all[post_all["cited_article"] == art]
    if min(len(pre), len(post)) < cfg["min_conversations"]:
        return {"estimate": np.nan, "se": np.nan}
    m_post, v_post, n_post = _mean_var(post[outcome])
    m_pre, v_pre, n_pre = _mean_var(pre[outcome])
    est, var = m_post - m_pre, v_post / n_post + v_pre / n_pre
    if design in ("did", "did_rtm"):
        controls = [a for a in post_all["cited_article"].unique() if a != art and a not in fixed_nearby]
        c_pre = pre_all[pre_all["cited_article"].isin(controls)][outcome]
        c_post = post_all[post_all["cited_article"].isin(controls)][outcome]
        if min(len(c_pre), len(c_post)) < cfg["min_conversations"]:
            return {"estimate": np.nan, "se": np.nan}
        cm_post, cv_post, cn_post = _mean_var(c_post)
        cm_pre, cv_pre, cn_pre = _mean_var(c_pre)
        est -= cm_post - cm_pre
        var += cv_post / cn_post + cv_pre / cn_pre
    return {"estimate": est, "se": float(np.sqrt(var))}


def estimate_all(conv: pd.DataFrame, fixes: list[dict], design: str, cfg: dict) -> pd.DataFrame:
    z = stats.norm.ppf(1 - cfg["alpha"] / 2)
    rows = []
    for f in fixes:
        # other articles fixed within the windows are not clean controls
        nearby = {
            g["article"]
            for g in fixes
            if g["workspace"] == f["workspace"]
            and g["fix_id"] != f["fix_id"]
            and abs(g["day"] - f["day"]) <= cfg["post_days"] + cfg["rtm_pre"][0]
        }
        r = estimate_fix(conv, f, design, cfg, nearby)
        se = r["se"]
        rows.append(
            {
                "fix_id": f["fix_id"],
                "design": design,
                "estimate": r["estimate"],
                "se": se,
                "low": r["estimate"] - z * se,
                "high": r["estimate"] + z * se,
            }
        )
    return pd.DataFrame(rows)


def pooled_effect(per_fix: pd.DataFrame, alpha: float) -> dict:
    """Inverse-variance weighted mean of the per-fix estimates (fixed effect)."""
    ok = per_fix.dropna(subset=["estimate", "se"])
    ok = ok[ok["se"] > 0]
    if ok.empty:
        return {"estimate": np.nan, "low": np.nan, "high": np.nan, "fixes": 0}
    wts = 1 / ok["se"] ** 2
    est = float((wts * ok["estimate"]).sum() / wts.sum())
    se = float(np.sqrt(1 / wts.sum()))
    z = stats.norm.ppf(1 - alpha / 2)
    return {"estimate": est, "low": est - z * se, "high": est + z * se, "fixes": int(len(ok))}


def minimum_detectable_effect(p: float, n_per_arm: float, alpha: float, power: float, sd: float | None = None) -> float:
    """Smallest true difference a two-arm test detects with the given power."""
    s = sd if sd is not None else np.sqrt(p * (1 - p))
    return float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * s * np.sqrt(2 / n_per_arm))
