# 0004: Topics from text, merged by the agent's citations

## Status

Accepted, 2026-11-21.

## Context

Keyword rules kept by hand cover only what someone thought of, in one language. Grouping by the article
the agent cited is accurate where an article exists, and blind exactly where it does not: a question with
no article is answered from whatever is closest, so it is spread over many citations.

## Decision

Cluster character n-gram TF-IDF (robust to typos and German compounds) into deliberately many clusters,
merge clusters that one article dominates, join the rest by text similarity, and flag a cluster as a
content gap when no article cites more than half of it.

## Consequences

- Citations are weak labels the product already logs; the method uses them without trusting them.
- A sentence-embedding model would separate paraphrases better; TF-IDF keeps this runnable on a laptop
  and in CI without model downloads.
