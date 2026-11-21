"""Runs all three questions on a set of seeds and writes reports/<set>.json.

Development seeds can be run as often as needed. Held-out seeds run once: the first run writes
reports/heldout.lock with the config fingerprint, and a second run refuses (see AGENTS.md).
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from rll import evaluate
from rll.config import ROOT, load_methods, load_seeds
from rll.world.sim import simulate

REPORTS = ROOT / "reports"
LOCK = REPORTS / "heldout.lock"


def config_fingerprint() -> str:
    h = hashlib.sha256()
    for name in ("world.yml", "topics.yml", "methods.yml", "gates.yml", "seeds.yml"):
        h.update((ROOT / "config" / name).read_bytes())
    return h.hexdigest()[:16]


def run_seed(seed: int) -> dict:
    m = load_methods()
    world = simulate(seed)
    topics = evaluate.grade_topics(world, m["topics"])
    res, _ = evaluate.grade_resolution(world, m["resolution"])
    per_fix, extra = evaluate.grade_lift(world, m["resolution"], m["lift"])
    truth = world.truth()
    outcome = world.observed().conversations["outcome"].to_numpy()
    solved = truth["true_resolved"].astype(bool).to_numpy()
    return {
        "seed": seed,
        "topics": topics.assign(seed=seed),
        "resolution": res.assign(seed=seed),
        "lift": per_fix.assign(seed=seed),
        "mde": extra["mde_median"],
        "facts": {
            "conversations": int(len(truth)),
            "fixes": len(world.fixes),
            "audits": int(len(world.audits)),
            "billed": int(truth["billable"].sum()),
            "billed_not_resolved": int((truth["billable"] & ~truth["true_resolved"].astype(bool)).sum()),
            "confirmed": int((outcome == "confirmed").sum()),
            "assumed_resolved": int(((outcome == "assumed") & solved).sum()),
            "assumed_unresolved": int(((outcome == "assumed") & ~solved).sum()),
            "escalated": int((outcome == "escalated").sum()),
            "true_resolution_rate": float(truth["true_resolved"].mean()),
            "billed_rate": float(truth["billable"].mean()),
        },
    }


def _r(x: float | np.floating | None, d: int = 4) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def summarize(parts: list[dict]) -> dict:
    m = load_methods()
    topics = pd.concat([p["topics"] for p in parts], ignore_index=True)
    res = pd.concat([p["resolution"] for p in parts], ignore_index=True)
    per_fix = pd.concat([p["lift"] for p in parts], ignore_index=True)
    facts: dict = {
        k: int(sum(p["facts"][k] for p in parts))
        for k in (
            "conversations",
            "fixes",
            "audits",
            "billed",
            "billed_not_resolved",
            "confirmed",
            "assumed_resolved",
            "assumed_unresolved",
            "escalated",
        )
    }
    facts["true_resolution_rate"] = float(np.mean([p["facts"]["true_resolution_rate"] for p in parts]))
    facts["billed_rate"] = float(np.mean([p["facts"]["billed_rate"] for p in parts]))

    t = topics.groupby("method").agg(
        ari=("ari", "mean"),
        nmi=("nmi", "mean"),
        assigned=("assigned", "mean"),
        clusters=("clusters", "mean"),
        gaps_found=("gap_found", "sum"),
        workspaces=("gap_found", "size"),
        false_gap_flags=("false_gap_flags", "sum"),
    )
    t_lang = topics.groupby(["method", "lang"])["ari"].mean().unstack()

    def res_row(g: pd.DataFrame) -> dict:
        has_ci = g["low"].notna().any()
        return {
            "mae": _r(g["error"].abs().mean()),
            "bias": _r(g["error"].mean()),
            "p90_abs_error": _r(g["error"].abs().quantile(0.9)),
            "coverage": _r(g["covered"].astype(float).mean()) if has_ci else None,
            "width": _r(g["width"].mean()) if has_ci else None,
            "units": int(len(g)),
        }

    lift_summary = evaluate.summarize_lift(per_fix, m["lift"]["alpha"])
    ppi = res[res["estimator"] == "ppi"]
    sample = per_fix[per_fix["design"].isin(["before_after_billed", "ab_holdout"])].dropna(subset=["estimate"])
    sample = pd.concat([g.sample(n=min(160, len(g)), random_state=0) for _, g in sample.groupby("design")])
    return {
        "facts": {k: _r(v) if isinstance(v, float) else int(v) for k, v in facts.items()},
        "topics": {
            meth: {**{k: _r(v) for k, v in row.items()}, "ari_en": _r(t_lang.loc[meth].get("en")), "ari_de": _r(t_lang.loc[meth].get("de"))}
            for meth, row in t.iterrows()
        },
        "resolution": {e: res_row(g) for e, g in res.groupby("estimator")},
        "resolution_units": [
            {
                "workspace": int(r.workspace),
                "block": int(r.block),
                "seed": int(r.seed),
                "true": _r(r.true_rate),
                "ppi": _r(r.estimate),
                "low": _r(r.low),
                "high": _r(r.high),
                "billed": _r(
                    res[
                        (res.seed == r.seed) & (res.workspace == r.workspace) & (res.block == r.block) & (res.estimator == "billing_proxy")
                    ]["estimate"].iloc[0]
                ),
            }
            for r in ppi.itertuples()
            if r.seed == parts[0]["seed"]
        ],
        "lift": {
            row["design"]: {k: (_r(v) if isinstance(v, float) else v) for k, v in row.items() if k != "design"}
            for row in lift_summary.to_dict("records")
        },
        "lift_points": [{"design": r.design, "estimate": _r(r.estimate), "truth": _r(r.true_effect)} for r in sample.itertuples()],
        "mde_median": _r(float(np.nanmedian([p["mde"] for p in parts]))),
    }


def run(set_name: str, workers: int = 3) -> dict:
    seeds = load_seeds()[set_name]
    fp = config_fingerprint()
    if set_name == "heldout":
        if LOCK.exists():
            raise SystemExit(f"held-out seeds already ran ({LOCK.read_text().strip()}); they run once. See AGENTS.md.")
        REPORTS.mkdir(exist_ok=True)
        LOCK.write_text(json.dumps({"ran_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "config_sha": fp, "seeds": seeds}) + "\n")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(run_seed, seeds))
    doc = {
        "set": set_name,
        "seeds": seeds,
        "config_sha": fp,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **summarize(parts),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"{set_name}.json").write_text(json.dumps(doc, indent=2) + "\n")
    return doc
