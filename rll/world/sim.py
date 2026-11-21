"""The simulated support world.

A world is a set of businesses (workspaces) whose customers write to an AI agent. The agent answers
with a help-centre article; whether the customer's problem is really solved is the world's hidden
truth. What the business sees is what the customer did next: confirmed, went quiet (an "assumed
resolution", billed like a confirmed one) or asked for more help; plus, a week later, whether the
customer came back about the same thing. A small weekly audit records the truth for a random sample.

Each Monday the content team picks an article to improve, mostly the one with the lowest
resolution rate last week, the status-quo process. That choice is made on a noisy dip, so the
next week tends to look better whatever the fix did: regression to the mean. The agent's model also
improves for everyone on one day, which a before/after comparison will credit to any fix nearby.

Every conversation after a fix has two potential outcomes, with the old and the new article, drawn
from the same random numbers. The full-rollout view shows the new one for everyone; the A/B view
shows a random half the old one for `holdout_days`. Both views share all other luck.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rll.config import ROOT, load_world

OBSERVED_COLUMNS = [
    "conv_id",
    "workspace",
    "vertical",
    "lang",
    "day",
    "weekday",
    "segment",
    "text",
    "cited_article",
    "answer_confidence",
    "outcome",
    "quiet_seconds",
    "customer_turns",
    "returned_7d",
    "arm",
]


@dataclass(frozen=True)
class Fix:
    fix_id: int
    workspace: int
    article: str
    day: int  # first day the new article is live
    selected_on_rate: float  # last week's observed resolution rate that triggered the pick
    reason: str  # "lowest_rate" or "requested"


@dataclass
class World:
    """Everything, including the truth. Methods never receive this; they get `Observed`."""

    seed: int
    conversations: pd.DataFrame  # observed columns + truth columns
    fixes: list[Fix]
    gaps: dict[int, str]  # workspace -> the topic with no article
    true_fix_effects: pd.DataFrame  # per fix: true change in resolution rate over the post window
    audits: pd.DataFrame  # conv_id, true_resolved for the audited sample
    config: dict

    def observed(self, design: str = "full_rollout") -> Observed:
        c = self.conversations
        if design == "ab_holdout":
            new = c["arm"].ne("old")
        elif design == "full_rollout":
            new = pd.Series(True, index=c.index)
        else:
            raise ValueError(f"unknown design {design}")
        view = c.copy()
        for col in ("outcome", "quiet_seconds", "returned_7d", "customer_turns", "answer_confidence"):
            view[col] = np.where(new, c[f"{col}__new"], c[f"{col}__old"])
        if design == "full_rollout":
            view["arm"] = None
        truth_resolved = np.where(new, c["resolved__new"], c["resolved__old"])
        audit = self.audits.merge(pd.DataFrame({"conv_id": c["conv_id"], "true_resolved": truth_resolved}), on="conv_id")
        return Observed(
            design=design,
            conversations=view[OBSERVED_COLUMNS].copy(),
            fixes=list(self.fixes),
            audit_labels=audit[["conv_id", "true_resolved"]].copy(),
            days=self.config["days"],
        )

    def truth(self, design: str = "full_rollout") -> pd.DataFrame:
        c = self.conversations
        new = c["arm"].ne("old") if design == "ab_holdout" else pd.Series(True, index=c.index)
        return pd.DataFrame(
            {
                "conv_id": c["conv_id"],
                "workspace": c["workspace"],
                "day": c["day"],
                "true_topic": c["true_topic"],
                "cited_article": c["cited_article"],
                "true_resolved": np.where(new, c["resolved__new"], c["resolved__old"]),
                "billable": np.where(new, c["outcome__new"], c["outcome__old"]) != "escalated",
            }
        )


@dataclass
class Observed:
    """What a method may use: conversations as logged, the fix log, and the weekly audit labels."""

    design: str
    conversations: pd.DataFrame
    fixes: list[Fix]
    audit_labels: pd.DataFrame
    days: int


def load_topics(path: Path | None = None) -> dict:
    return yaml.safe_load((path or ROOT / "config" / "topics.yml").read_text())


def _fill(template: str, slots: dict, rng: np.random.Generator) -> str:
    def pick(m: re.Match[str]) -> str:
        values = slots[m.group(1)]
        return str(values[rng.integers(len(values))])

    text = re.sub(r"\{(\w+)\}", pick, template)
    return re.sub(r"\s+", " ", text).strip()


def _typo(text: str, rng: np.random.Generator) -> str:
    if len(text) < 8:
        return text
    i = int(rng.integers(1, len(text) - 2))
    if rng.random() < 0.5:
        return text[:i] + text[i + 1] + text[i] + text[i + 2 :]
    return text[:i] + text[i + 1 :]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(seed: int, config: dict | None = None, topics: dict | None = None) -> World:
    cfg = config or load_world()
    cat = topics or load_topics()
    rng = np.random.default_rng(seed)
    verticals = list(cat["verticals"])
    res = cfg["resolution"]
    beh = cfg["behaviour"]
    days = cfg["days"]

    # --- workspaces ------------------------------------------------------------------------------
    spaces = []
    for w in range(cfg["workspaces"]):
        vertical = verticals[w % len(verticals)]
        lang = cfg["languages"][w % len(cfg["languages"])]
        names = list(cat["verticals"][vertical])
        mix = rng.dirichlet(np.full(len(names), 3.0))
        coverage = np.clip(rng.normal(res["coverage_start"]["mean"], res["coverage_start"]["sd"], len(names)), 0.05, 0.95)
        volume = float(np.exp(np.log(cfg["daily_volume"]["median"]) + rng.normal(0, cfg["daily_volume"]["spread"])))
        # one topic has no article yet: the agent always cites something else (a content gap)
        gap = str(names[int(rng.integers(len(names)))])
        ask_shift = float(rng.normal(0, cfg["behaviour"]["ask_more_spread"]))
        spaces.append(
            {
                "vertical": vertical,
                "lang": lang,
                "topics": names,
                "mix": mix,
                "gap": gap,
                "articles": [t for t in names if t != gap],
                "ask_shift": ask_shift,
                "coverage": dict(zip(names, coverage, strict=True)),
                "volume": volume,
            }
        )

    # --- incidents ---------------------------------------------------------------------------------
    inc = cfg["incidents"]
    incidents: list[tuple[int, str, int, int, float, float]] = []
    for w, s in enumerate(spaces):
        n = rng.poisson(inc["per_workspace_month"] * days / 30)
        for _ in range(n):
            start = int(rng.integers(0, days))
            length = int(rng.integers(inc["length_days"][0], inc["length_days"][1] + 1))
            incidents.append(
                (
                    w,
                    str(rng.choice(s["topics"])),
                    start,
                    start + length,
                    float(rng.uniform(*inc["volume_multiplier"])),
                    float(rng.uniform(*inc["resolution_penalty"])),
                )
            )

    seg_names = list(res["segment_mix"])
    seg_p = np.array([res["segment_mix"][k] for k in seg_names])
    fx = cfg["fixes"]
    weekly = np.array(cfg["weekly_shape"])

    rows: list[pd.DataFrame] = []
    fixes: list[Fix] = []
    # article quality before/after each fix and the conversations in its holdout window
    pending: dict[tuple[int, str], tuple[float, float, int, int]] = {}  # (w, art) -> (old, new, start, fix_id)
    last_fix_day: dict[tuple[int, str], int] = {}
    conv_id = 0

    for day in range(days):
        weekday = day % 7
        # Monday: the content team picks an article to fix from last week's observed numbers
        if weekday == fx["weekday"] and day >= 7:
            recent = pd.concat([r for r in rows[-7 * len(spaces) :]], ignore_index=True) if rows else None
            for w, s in enumerate(spaces):
                if recent is None:
                    break
                wk = recent[(recent["workspace"] == w)]
                stats = wk.assign(ok=wk["outcome__new"] != "escalated").groupby("cited_article")["ok"].agg(["mean", "size"])
                stats = stats[stats["size"] >= fx["min_conversations"]]
                stats = stats[[last_fix_day.get((w, a), -999) <= day - fx["cooldown_days"] for a in stats.index]]
                if stats.empty:
                    continue
                if rng.random() < fx["pick_lowest_rate"]:
                    art, reason = str(stats["mean"].idxmin()), "lowest_rate"
                else:
                    art, reason = str(rng.choice(stats.index)), "requested"
                old = s["coverage"][art]
                new = float(np.clip(old + rng.normal(fx["effect"]["mean"], fx["effect"]["sd"]), 0.02, 0.99))
                fix = Fix(len(fixes), w, art, day, float(stats.loc[art, "mean"]), reason)
                fixes.append(fix)
                pending[(w, art)] = (old, new, day, fix.fix_id)
                s["coverage"][art] = new
                last_fix_day[(w, art)] = day

        for w, s in enumerate(spaces):
            lam = s["volume"] * weekly[weekday]
            vol = s["mix"] * lam
            mult = np.ones(len(s["topics"]))
            pen = np.zeros(len(s["topics"]))
            for iw, itopic, a, b, vm, rp in incidents:
                if iw == w and a <= day < b:
                    k = s["topics"].index(itopic)
                    mult[k] *= vm
                    pen[k] += rp
            counts = rng.poisson(vol * mult)
            n = int(counts.sum())
            if n == 0:
                continue
            tidx = np.repeat(np.arange(len(s["topics"])), counts)
            true_topic = np.array(s["topics"])[tidx]
            # the agent cites the right article, or occasionally another one
            mis = (rng.random(n) < res["misretrieval"]) | (true_topic == s["gap"])
            other = np.array(s["articles"])[rng.integers(0, len(s["articles"]), n)]
            cited = np.where(mis, other, true_topic)
            segment = np.array(seg_names)[rng.choice(len(seg_names), n, p=seg_p)]
            diff = np.array([cat["verticals"][s["vertical"]][t]["difficulty"] for t in true_topic])
            seg_eff = np.array([res["segment"][g] for g in segment])
            upgrade = res["model_upgrade"]["effect"] if day >= res["model_upgrade"]["day"] else 0.0
            noise = rng.normal(0, res["conversation_noise"], n)

            cov_new = np.array([s["coverage"][a] for a in cited])
            cov_old = cov_new.copy()
            arm = np.full(n, None, dtype=object)
            u_arm = rng.random(n)
            for i, a in enumerate(cited):
                p = pending.get((w, a))
                if p and p[2] <= day < p[2] + fx["holdout_days"]:
                    cov_old[i] = p[0]
                    arm[i] = "old" if u_arm[i] < 0.5 else "new"
            # a wrong article barely helps
            cov_new = np.where(mis, cov_new * 0.2, cov_new)
            cov_old = np.where(mis, cov_old * 0.2, cov_old)
            base = res["base"] - res["difficulty_weight"] * diff + seg_eff + upgrade - pen[tidx] + noise
            p_new = _sigmoid(base + res["coverage_weight"] * cov_new)
            p_old = _sigmoid(base + res["coverage_weight"] * cov_old)
            u_res, u_beh, u_ret, u_conf = rng.random(n), rng.random(n), rng.random(n), rng.normal(0, 1, n)
            quiet_z = rng.normal(0, 1, n)
            ask_more = np.clip(np.array([beh["unresolved_ask_more"][g] for g in segment]) + s["ask_shift"], 0.05, 0.95)

            frame = {
                "conv_id": np.arange(conv_id, conv_id + n),
                "workspace": w,
                "vertical": s["vertical"],
                "lang": s["lang"],
                "day": day,
                "weekday": weekday,
                "segment": segment,
                "true_topic": true_topic,
                "cited_article": cited,
                "arm": arm,
                "p_true__new": p_new,
                "p_true__old": p_old,
            }
            for version, p_true, cov in (("new", p_new, cov_new), ("old", p_old, cov_old)):
                resolved = u_res < p_true
                r = beh["resolved"]
                outcome = np.where(
                    resolved,
                    np.where(u_beh < r["confirm"], "confirmed", np.where(u_beh < r["confirm"] + r["quiet"], "assumed", "escalated")),
                    np.where(u_beh < ask_more, "escalated", "assumed"),
                )
                rt = beh["return_7d"]
                p_ret = np.where(
                    outcome == "confirmed", rt["resolved_confirm"], np.where(resolved, rt["resolved_quiet"], rt["unresolved_quiet"])
                )
                returned = (outcome != "escalated") & (u_ret < p_ret)
                q = beh["quiet_seconds"]
                mu = np.where(resolved, q["resolved"][0], q["unresolved"][0])
                sd = np.where(resolved, q["resolved"][1], q["unresolved"][1])
                quiet = np.where(outcome == "assumed", np.exp(mu + sd * quiet_z), np.nan)
                ac = cfg["answer_confidence"]
                conf = np.clip(0.3 + ac["coverage_weight"] * cov + ac["truth_weight"] * resolved + ac["noise"] * u_conf, 0, 1)
                turns = 1 + (outcome == "escalated") + (u_beh * 3 * (~resolved)).astype(int)
                frame[f"resolved__{version}"] = resolved
                frame[f"outcome__{version}"] = outcome
                frame[f"returned_7d__{version}"] = returned
                frame[f"quiet_seconds__{version}"] = quiet
                frame[f"answer_confidence__{version}"] = conf
                frame[f"customer_turns__{version}"] = turns
            # text: what the customer wrote first
            texts = []
            for t in true_topic:
                tpl = cat["verticals"][s["vertical"]][t][s["lang"]]
                txt = _fill(tpl[rng.integers(len(tpl))], cat["slots"], rng)
                texts.append(_typo(txt, rng) if rng.random() < 0.15 else txt)
            frame["text"] = texts
            frame["is_gap"] = true_topic == s["gap"]
            rows.append(pd.DataFrame(frame))
            conv_id += n

    conv = pd.concat(rows, ignore_index=True)
    conv["arm"] = conv["arm"].where(conv["arm"].isin(["old", "new"]), None)
    # returns are only known a week later; the last week has none yet
    for v in ("new", "old"):
        conv[f"returned_7d__{v}"] = np.where(conv["day"] < days - 7, conv[f"returned_7d__{v}"].astype(float), np.nan)

    # weekly audit: a random sample per workspace and week
    audits = []
    per = cfg["audit"]["per_workspace_week"]
    for (_w, _wk), g in conv.groupby([conv["workspace"], conv["day"] // 7]):
        take = min(per, len(g))
        audits.append(g.sample(n=take, random_state=int(rng.integers(1 << 31)))[["conv_id"]])
    audits_df = pd.concat(audits, ignore_index=True)

    effects = _true_effects(conv, fixes, fx["holdout_days"])
    return World(
        seed=seed,
        conversations=conv,
        fixes=fixes,
        gaps={w: s["gap"] for w, s in enumerate(spaces)},
        true_fix_effects=effects,
        audits=audits_df,
        config=cfg,
    )


def _true_effects(conv: pd.DataFrame, fixes: list[Fix], window: int) -> pd.DataFrame:
    """The truth each design is graded against: the mean change in resolution probability the fix
    caused for the conversations citing that article in the `window` days after it went live."""
    out = []
    for f in fixes:
        m = (conv["workspace"] == f.workspace) & (conv["cited_article"] == f.article) & conv["day"].between(f.day, f.day + window - 1)
        g = conv[m]
        effect = float((g["p_true__new"] - g["p_true__old"]).mean()) if len(g) else 0.0
        out.append(
            {
                "fix_id": f.fix_id,
                "workspace": f.workspace,
                "article": f.article,
                "day": f.day,
                "true_effect": effect,
                "post_conversations": int(len(g)),
                "reason": f.reason,
            }
        )
    return pd.DataFrame(out)
