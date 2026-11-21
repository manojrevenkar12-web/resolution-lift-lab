# Working on this repository

## Fast feedback

- `pytest -q` runs in about 5 s on small worlds.
- `python -m rll.cli run dev && python -m rll.cli gate dev && python -m rll.cli report` runs the three
  development seeds (about 3 minutes on 3 cores) and regenerates REPORT.md, the figures and the README block.
- `cd web && npm test && npm run typecheck && npm run lint` for the dashboard.

## Invariants

Each has a test or a CI step.

- **The truth stays in the world.** Methods get `Observed` (logged conversations, the fix log, the weekly
  audit labels), never `World`. Truth is joined in `rll/evaluate.py` after a method has answered
  (`tests/test_world.py`, `tests/test_boundaries.py`).
- **The world does not read the methods' settings**, and methods never import `rll.world`
  (`tests/test_boundaries.py`).
- **Same luck for every design.** Both views of a world (full rollout, A/B holdout) come from the same
  random numbers; potential outcomes share them (`test_both_designs_share_all_luck_outside_the_holdout`).
- **A model never grades its own workspace.** Resolution models are cross-fitted by workspace, so the
  prediction-powered correction uses out-of-sample errors (`test_cross_fit_never_trains_on_a_workspace_own_audits`).
- **Settings live in `config/`**: the world in `world.yml` and `topics.yml`, methods in `methods.yml`,
  thresholds in `gates.yml`, seeds in `seeds.yml`.
- **Reports are generated, never edited.** CI regenerates REPORT.md and the README block and fails on a diff.

## Held-out seeds

- `config/seeds.yml` lists them. `python -m rll.cli run heldout` writes `reports/heldout.lock` before it
  runs and refuses to run again. CI checks the lock was committed once.
- Gates were set from development seeds before the held-out run. Never loosen a gate after it.
- If a change needs new held-out evidence, add new seeds under a new name and record that in an ADR.

## Data

Everything is simulated: workspaces, customers, texts and the content team. No real conversations.
