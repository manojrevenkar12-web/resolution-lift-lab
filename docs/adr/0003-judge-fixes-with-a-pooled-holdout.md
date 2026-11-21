# 0003: Judge content fixes with a pooled holdout, not before/after

## Status

Accepted, 2026-11-21.

## Context

The content team fixes the article that looked worst last week. Part of last week was bad luck, so the
next weeks look better whatever the fix did. A model upgrade on one day adds to that. A per-article
two-week A/B test is unbiased, but with one article's traffic it can only detect changes of tens of
points, while fixes move a few.

## Decision

- Hold out half of the conversations on the old article for 14 days after every fix (same luck in both
  arms), and combine the per-fix differences by inverse-variance weighting. That answers the question
  the product needs: what does a recommended fix do on average.
- When a holdout is not possible, use difference-in-differences whose "before" window skips the week
  that triggered the fix. It removes most of the regression to the mean but is only available for
  fixes with three clean weeks of history.
- Do not use before/after for decisions.

## Consequences

- A holdout delays part of each fix's benefit by two weeks. With fixes worth a few points, that costs
  little next to shipping a programme whose effect is overstated by half.
- Individual fixes stay unjudged. Ranking fixes would need a model of fix effects, not more tests.
