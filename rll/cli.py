"""python -m rll.cli run dev | run heldout | report | gate"""

from __future__ import annotations

import argparse
import json

from rll import report, run
from rll.config import load_gates


def main() -> None:
    ap = argparse.ArgumentParser(prog="rll")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a seed set and write reports/<set>.json")
    r.add_argument("set", choices=["dev", "heldout"])
    r.add_argument("--workers", type=int, default=3)
    sub.add_parser("report", help="write REPORT.md, figures and the README results block")
    g = sub.add_parser("gate", help="fail if a report misses a threshold in config/gates.yml")
    g.add_argument("set", choices=["dev", "heldout"], nargs="?", default="dev")
    args = ap.parse_args()
    if args.cmd == "run":
        doc = run.run(args.set, workers=args.workers)
        print(json.dumps({k: doc[k] for k in ("set", "seeds", "config_sha")}))
    elif args.cmd == "report":
        report.write()
    else:
        failures = report.gate(json.loads((run.REPORTS / f"{args.set}.json").read_text()), load_gates())
        for f in failures:
            print(f"GATE FAILED: {f}")
        if failures:
            raise SystemExit(1)
        print(f"gate passed on {args.set}")


if __name__ == "__main__":
    main()
