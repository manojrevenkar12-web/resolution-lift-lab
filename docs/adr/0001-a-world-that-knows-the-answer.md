# 0001: Grade every method against a simulated world that knows the answer

## Status

Accepted, 2026-11-21.

## Context

The three questions here (were resolutions real, did a fix work, which questions have no article) are
questions about things production data never shows directly. A real support log has no column for
"this customer gave up" or "this fix caused a 3-point lift". So any method can look good on real data.

## Decision

- A simulated world (`rll/world/sim.py`) generates businesses, customers, texts, the agent's answers
  and citations, customer behaviour, a delayed return signal, a weekly audit, and a content team that
  fixes the worst-looking article each Monday. It records the truth: whether each conversation was
  solved, both potential outcomes after a fix, and the one topic per business with no article.
- Methods see only what a product would log. They are graded in `rll/evaluate.py`.
- The world is built to contain the traps a real analysis falls into: customers who give up go quiet
  (billed as resolved); fixes follow a bad week (regression to the mean); the agent's model improves for
  everyone on one day; businesses differ in how often unhappy customers ask again.

## Consequences

- Numbers are only as good as the world. They show which method is right *given these mechanisms*,
  not how large the effects are at any real company.
- The first thing to do with real data is to check the mechanisms (is there a regression-to-the-mean
  dip before fixes? do returners differ from the audit?), then rerun the comparison.
