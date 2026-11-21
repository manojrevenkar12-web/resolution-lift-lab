# 0002: Estimate true resolutions with prediction-powered inference

## Status

Accepted, 2026-11-21.

## Context

The billed rate counts customers who went quiet for a day. A weekly audit knows the truth for about 70
conversations per business and month: unbiased, but its interval is about 20 points wide. A classifier
on behaviour signals (outcome, time to leave, 7-day return, answer confidence) is precise, but it can be
biased for a business whose customers behave differently, and it has no honest interval.

## Decision

Prediction-powered inference (Angelopoulos et al., *Science* 2023): the classifier's mean over every
conversation, plus its mean error on the audited ones. The classifier is cross-fitted by workspace, so
its error on a business's audits is measured out of sample.

## Consequences

- On these worlds the classifier alone is even closer on average, but it gives no interval and its
  error is not guaranteed. Use it for dashboards; use PPI for anything a customer may dispute, such as
  a bill.
- The audit sample must stay random. If reviewers pick conversations, the correction is biased.
