"""Health check for a running grid: progress, validity, failures, stalls, sanity.

  python scripts/health.py results/main/runs/eval [--expect 24]

Prints one block the watcher can read; exits 2 when something looks wrong
(invalid rounds, failed calls, FAIL/STOPPED lines, or no new round in 20 min).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collusion_dial.metrics.collusion import load_run, round_frame, summarize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--expect", type=int, default=None, help="expected number of runs")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--grid-log", default=None)
    a = ap.parse_args()
    logs = Path(a.logs)
    problems = []
    complete, partial, newest = 0, 0, 0.0
    rounds_total = invalid = calls = failed = 0
    per_cond = {}
    for meta_path in sorted(logs.glob("*/meta.json")):
        rd = meta_path.parent
        rj = rd / "rounds.jsonl"
        if not rj.exists():
            partial += 1
            continue
        newest = max(newest, rj.stat().st_mtime)
        meta, rounds = load_run(rd)
        n_expected = len(meta["schedule"])
        rounds_total += len(rounds)
        invalid += sum(not r["valid"] for r in rounds)
        cs = [c for r in rounds for c in r["calls"]]
        calls += len(cs)
        failed += sum(not c["ok"] for c in cs)
        if len(rounds) >= n_expected:
            complete += 1
            s = summarize(round_frame(meta, rounds))
            per_cond.setdefault(meta["condition"], []).append(
                (s.get("collusion_index"), sum(len(r["chat"]) for r in rounds)))
        else:
            partial += 1
    print(f"runs: complete={complete} in_progress={partial}" + (f" expected={a.expect}" if a.expect else ""))
    print(f"rounds={rounds_total} invalid={invalid} calls={calls} failed_calls={failed}")
    if newest:
        age = (time.time() - newest) / 60
        print(f"last round written {age:.1f} min ago")
        if age > 20 and (a.expect is None or complete < a.expect):
            problems.append(f"no new round for {age:.0f} min")
    for c, v in sorted(per_cond.items()):
        idx = [x for x, _ in v if x is not None and x == x]
        msgs = [m for _, m in v]
        print(f"  {c:<20} n={len(v):>2} collusion_index mean={sum(idx) / len(idx) if idx else float('nan'):.2f} "
              f"messages mean={sum(msgs) / len(msgs):.1f}")
    if a.grid_log and Path(a.grid_log).exists():
        bad = [l.strip() for l in open(a.grid_log, encoding="utf-8", errors="replace")
               if l.startswith("FAIL") or l.startswith("STOPPED")]
        if bad:
            problems.append("grid log: " + "; ".join(bad[:3]))
    if rounds_total and invalid / rounds_total > 0.02:
        problems.append(f"invalid rounds {invalid}/{rounds_total}")
    if calls and failed / calls > 0.02:
        problems.append(f"failed calls {failed}/{calls}")
    # design sanity on complete runs: positives should carry messages (explicit) or not (tacit)
    for c, v in per_cond.items():
        if c == "explicit_collusion" and len(v) >= 2 and sum(m for _, m in v) / len(v) < 6:
            problems.append("explicit_collusion runs have almost no messages")
        if c in ("tacit_collusion", "honest_nomsg") and any(m > 0 for _, m in v):
            problems.append(f"{c} has messages (messaging should be off)")
    if problems:
        print("PROBLEMS: " + " | ".join(problems))
        sys.exit(2)
    print("OK")


if __name__ == "__main__":
    main()
