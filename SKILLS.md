# Repeatable tasks

## Try a new estimator

1. Add it to `rll/resolution.py` (or `rll/lift.py`), reading only `Observed` columns.
2. Add its name to `ESTIMATORS` / `DESIGNS`; `rll/evaluate.py` grades it against the truth.
3. Write a test on a synthetic case where the right answer is known (see `tests/test_lift.py`).
4. `python -m rll.cli run dev && python -m rll.cli report` and read the tables.

## Change the world

Edit `config/world.yml` or `config/topics.yml`. Keep a reason in a comment; the world is the claim
every number rests on. Rerun dev. Held-out results recorded under the old world stay as they are.

## Tighten a gate

Edit `config/gates.yml` and run `python -m rll.cli gate dev`. Gates only ever get stricter.

## Refresh the dashboard

`cd web && npm run dev` copies `reports/*.json` into `public/` and serves the page.
