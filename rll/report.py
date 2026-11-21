"""Generated reports: REPORT.md, figures in docs/figures, and the results block in README.md.
Nothing here is edited by hand; run `python -m rll.cli report` after a run."""

from __future__ import annotations

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rll.config import ROOT  # noqa: E402

REPORTS = ROOT / "reports"
FIGS = ROOT / "docs" / "figures"

TOPIC_NAMES = {
    "keyword_rules": "Keyword rules (status quo)",
    "by_citation": "One topic per cited article",
    "word_kmeans": "Word TF-IDF + k-means",
    "char_kmeans": "Character n-grams + k-means",
    "char_kmeans_cited": "Character n-grams, merged by citations (this repo)",
}
RES_NAMES = {
    "billing_proxy": "Billed rate (status quo)",
    "return_adjusted": "Billed, minus customers who came back",
    "audit_only": "Weekly audit alone",
    "model_only": "Model alone",
    "ppi": "Prediction-powered inference (this repo)",
}
LIFT_NAMES = {
    "before_after_billed": "Billed rate, week before vs two weeks after (status quo)",
    "before_after_model": "Model estimate, before vs after",
    "did": "Difference-in-differences",
    "did_rtm": "Difference-in-differences, skipping the trigger week",
    "ab_holdout": "A/B holdout, pooled over fixes (this repo)",
}


def pct(x: float | None, d: int = 1) -> str:
    return "–" if x is None else f"{x * 100:.{d}f}%"


def pts(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:+.1f} pts"


def load(name: str) -> dict | None:
    p = REPORTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def tables(doc: dict) -> str:
    f, t, r, lf = doc["facts"], doc["topics"], doc["resolution"], doc["lift"]
    lines = [
        f"Seeds {', '.join(map(str, doc['seeds']))} · {f['conversations']:,} conversations · "
        f"{f['fixes']:,} content fixes · {f['audits']:,} audited conversations · config `{doc['config_sha']}`",
        "",
    ]
    lines += [
        "**How many billed resolutions were real** (true resolution rate per workspace and 4 weeks)",
        "",
        "| Estimator | Mean abs. error | Bias | 95% interval holds | Interval width |",
        "|---|--:|--:|--:|--:|",
    ]
    for k in ("billing_proxy", "return_adjusted", "audit_only", "model_only", "ppi"):
        v = r[k]
        lines.append(
            f"| {RES_NAMES[k]} | {pts(v['mae'])[1:]} | {pts(v['bias'])} | {pct(v['coverage'], 0)} | "
            f"{pts(v['width'])[1:] if v['width'] else '–'} |"
        )
    lines += [
        "",
        "**Did a content fix raise resolutions** (true effect: change in resolution probability over 14 days)",
        "",
        "| Design | Average effect claimed (95% CI) | True average | Fixes measured | Per-fix bias |",
        "|---|--:|--:|--:|--:|",
    ]
    for k in ("before_after_billed", "before_after_model", "did", "did_rtm", "ab_holdout"):
        v = lf[k]
        lines.append(
            f"| {LIFT_NAMES[k]} | {pts(v['pooled'])} ({pts(v['pooled_low'])} to {pts(v['pooled_high'])}) | "
            f"{pts(v['true_mean'])} | {v['estimated']} of {v['fixes']} | {pts(v['bias'])} |"
        )
    lines += [
        "",
        "**What customers write about, and which questions have no article**",
        "",
        "| Method | Agreement with true topics (ARI) | English / German | Conversations assigned | Content gaps found | False gap flags |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for k in ("keyword_rules", "by_citation", "word_kmeans", "char_kmeans", "char_kmeans_cited"):
        v = t[k]
        lines.append(
            f"| {TOPIC_NAMES[k]} | {v['ari']:.2f} | {v['ari_en']:.2f} / {v['ari_de']:.2f} | {pct(v['assigned'], 0)} | "
            f"{int(v['gaps_found'])} of {int(v['workspaces'])} | {int(v['false_gap_flags'])} |"
        )
    return "\n".join(lines)


def headline(doc: dict) -> str:
    f, r, lf, t = doc["facts"], doc["resolution"], doc["lift"], doc["topics"]
    share = f["billed_not_resolved"] / f["billed"]
    naive, ab = lf["before_after_billed"], lf["ab_holdout"]
    return "\n".join(
        [
            f"- **{pct(share)} of billed resolutions were not real.** Billed rate {pct(f['billed_rate'])}, true rate "
            f"{pct(f['true_resolution_rate'])}. Prediction-powered inference recovers the true rate within "
            f"{pts(r['ppi']['mae'])[1:]} on average with a 95% interval that held {pct(r['ppi']['coverage'], 0)} of the time, "
            f"from about 70 audited conversations per workspace and month.",
            f"- **The status-quo before/after view overstates what content fixes do by "
            f"{(naive['pooled'] / naive['true_mean'] - 1) * 100:.0f}%** "
            f"({pts(naive['pooled'])} claimed vs {pts(naive['true_mean'])} true): "
            f"the article picked was the one that had a bad week. A pooled A/B holdout gets {pts(ab['pooled'])} "
            f"against a true {pts(ab['true_mean'])}.",
            f"- **One article's two-week A/B can only detect a change of about {pts(doc['mde_median'])[1:]}**, while a typical fix "
            f"moves {pts(ab['true_mean'])[1:]}. Judge the Recommendations feature with a pooled holdout, not fix by fix.",
            f"- **Content gaps:** clustering text and merging clusters where the agent's citations agree found "
            f"{int(t['char_kmeans_cited']['gaps_found'])} of {int(t['char_kmeans_cited']['workspaces'])} questions with no article "
            f"(topic agreement ARI {t['char_kmeans_cited']['ari']:.2f}); hand-kept keyword rules reach ARI "
            f"{t['keyword_rules']['ari_de']:.2f} on German conversations.",
        ]
    )


def figures(doc: dict) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    ink, muted, accent, warn = "#1d2433", "#7a8194", "#2f6fdb", "#d1495b"
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": muted,
            "axes.labelcolor": ink,
            "xtick.color": muted,
            "ytick.color": muted,
        }
    )
    # pooled effect by design
    order = ["before_after_billed", "before_after_model", "did", "did_rtm", "ab_holdout"]
    lf = doc["lift"]
    fig, ax = plt.subplots(figsize=(8.6, 3.3))
    for i, k in enumerate(order):
        v = lf[k]
        color = accent if k == "ab_holdout" else (warn if abs(v["pooled"] - v["true_mean"]) > 0.01 else muted)
        ax.errorbar(
            v["pooled"] * 100,
            i,
            xerr=[[(v["pooled"] - v["pooled_low"]) * 100], [(v["pooled_high"] - v["pooled"]) * 100]],
            fmt="o",
            color=color,
            capsize=3,
            lw=2,
            ms=6,
        )
    truth = lf["ab_holdout"]["true_mean"] * 100
    ax.axvline(truth, color=ink, lw=1, ls="--")
    ax.text(truth, -0.75, " true average", color=ink, va="bottom", fontsize=9)
    ax.set_yticks(range(len(order)), [LIFT_NAMES[k].replace(" (this repo)", "").replace(" (status quo)", "") for k in order])
    ax.set_xlabel("Average effect of a content fix on resolution rate, percentage points (95% CI)")
    ax.set_ylim(len(order) - 0.5, -1.0)
    fig.savefig(FIGS / "fix-effect-by-design.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    # billed vs true vs ppi per workspace-month
    units = doc["resolution_units"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for i, u in enumerate(units):
        ax.plot([i, i], [u["low"] * 100, u["high"] * 100], color=accent, alpha=0.35, lw=3)
        ax.plot(i, u["ppi"] * 100, "o", color=accent, ms=3)
        ax.plot(i, u["true"] * 100, "_", color=ink, ms=9, mew=2)
        ax.plot(i, u["billed"] * 100, "x", color=warn, ms=4)
    ax.set_xticks([])
    ax.set_xlabel("Workspace × 4-week block (one seed)")
    ax.set_ylabel("Resolution rate, %")
    ax.plot([], [], "x", color=warn, label="billed")
    ax.plot([], [], "_", color=ink, mew=2, label="true")
    ax.plot([], [], "o", color=accent, label="PPI estimate and 95% interval")
    ax.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.savefig(FIGS / "billed-vs-true.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def write() -> None:
    dev, held = load("dev"), load("heldout")
    if dev is None:
        raise SystemExit("run `python -m rll.cli run dev` first")
    main = held or dev
    figures(main)
    body = ["# Results", "", "Generated by `python -m rll.cli report`. Do not edit by hand.", ""]
    if held:
        lock = json.loads((REPORTS / "heldout.lock").read_text())
        body += [f"## Held-out seeds (first and only run, {lock['ran_at']})", "", headline(held), "", tables(held), ""]
    body += ["## Development seeds", "", headline(dev), "", tables(dev), ""]
    (ROOT / "REPORT.md").write_text("\n".join(body))
    readme = ROOT / "README.md"
    if readme.exists():
        label = "held-out seeds, run once" if held else "development seeds"
        block = (
            f"<!-- results:start -->\n*{label}; generated {main['generated_at']} by `python -m rll.cli report`*\n\n"
            f"{headline(main)}\n\n{tables(main)}\n<!-- results:end -->"
        )
        text = readme.read_text()
        readme.write_text(re.sub(r"<!-- results:start -->.*?<!-- results:end -->", lambda _m: block, text, flags=re.S))


def gate(doc: dict, gates: dict) -> list[str]:
    out = []
    t = doc["topics"]["char_kmeans_cited"]
    if t["ari"] < gates["topics"]["ari_min"]:
        out.append(f"topic ARI {t['ari']:.3f} < {gates['topics']['ari_min']}")
    if t["gaps_found"] / t["workspaces"] < gates["topics"]["gaps_found_min"]:
        out.append(f"content gaps found {t['gaps_found']}/{t['workspaces']}")
    if t["false_gap_flags"] / t["workspaces"] > gates["topics"]["false_flags_per_workspace_max"]:
        out.append(f"false gap flags {t['false_gap_flags']} over {t['workspaces']} workspaces")
    p = doc["resolution"]["ppi"]
    if p["coverage"] < gates["resolution"]["ppi_coverage_min"]:
        out.append(f"PPI interval coverage {p['coverage']}")
    if p["mae"] > gates["resolution"]["ppi_mae_max"]:
        out.append(f"PPI mean abs. error {p['mae']}")
    ab = doc["lift"]["ab_holdout"]
    if abs(ab["pooled"] - ab["true_mean"]) > gates["lift"]["pooled_error_max"]:
        out.append(f"pooled A/B effect {ab['pooled']:.4f} vs true {ab['true_mean']:.4f}")
    if not ab["pooled_low"] <= ab["true_mean"] <= ab["pooled_high"]:
        out.append("pooled A/B interval misses the true average")
    return out
