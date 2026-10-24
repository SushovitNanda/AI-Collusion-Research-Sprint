"""Progress of the OpenRouter panel pipeline: done / left / time to completion.

  python scripts/panel_progress.py [results/model_panel]

Reads progress.txt (phase markers written by scripts/run_openrouter.sh), the run
folders (rounds written) and the monitor outputs (judge files written), and
extrapolates each phase from its own rate.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/model_panel")
MODELS = ["or_gptoss", "or_gemini", "or_deepseek", "or_llama", "or_mistral", "or_haiku"]
PLAN_RUNS = {"panel": 7, "panel_more": 8}            # runs per model, by grid plan name
PHASES = (("panel", "panel_s1"), ("panel_more", "panel_s23"))   # (plan, runs sub-folder)
VIEWS = 5                                            # judge files per run
ROUNDS = 12


def parse_prog():
    marks = {}
    p = OUT / "progress.txt"
    if not p.exists():
        return marks
    for line in p.read_text().splitlines():
        m = re.match(r"(\S+ \S+) (.*)", line)
        if m:
            marks[m.group(2)] = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
    return marks


def agent_status(folder: Path, expect_runs: int):
    done = partial = 0
    rounds = 0
    errors = 0
    newest = 0.0
    for meta in folder.glob("*/meta.json"):
        rd = meta.parent
        rj = rd / "rounds.jsonl"
        n = 0
        if rj.exists():
            newest = max(newest, rj.stat().st_mtime)
            for line in rj.open(encoding="utf8"):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n += 1
                errors += sum(1 for c in r["calls"] if not c["ok"])
        rounds += n
        if n >= ROUNDS:
            done += 1
        else:
            partial += 1
    return dict(done=done, partial=partial, rounds=rounds, total_rounds=expect_runs * ROUNDS,
                failed_calls=errors, newest=newest)


def monitor_status(folder: Path):
    runs = list(folder.glob("*/meta.json"))
    files = 0
    for meta in runs:
        md = meta.parent / "monitor"
        files += len(list(md.glob("glm__*.json"))) if md.exists() else 0
    return dict(runs=len(runs), judge_files=files, expected=len(runs) * VIEWS)


def fmt(sec):
    if sec is None:
        return "?"
    sec = max(0, int(sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main():
    now = time.time()
    marks = parse_prog()
    print(f"panel progress at {datetime.now():%H:%M:%S}")
    if not marks:
        print("  pipeline not started (no progress.txt)")
        return
    start = marks.get("PIPELINE START")
    print(f"  elapsed since start: {fmt(now - start) if start else '?'}")
    for plan, sub in PHASES:
        folder = OUT / "runs" / sub
        expect = PLAN_RUNS[plan] * len(MODELS)
        st = agent_status(folder, expect)
        t0 = marks.get(f"AGENTS START plan={plan} out={folder.as_posix()}")
        t1 = marks.get(f"AGENTS DONE plan={plan} out={folder.as_posix()}")
        if t0 is None:
            print(f"  agents {plan}: not started ({expect} runs planned)")
            continue
        eta = None
        if t1 is None and st["rounds"] > 0:
            rate = st["rounds"] / max(1, now - t0)
            eta = (st["total_rounds"] - st["rounds"]) / rate if rate > 0 else None
        stall = (now - st["newest"]) / 60 if st["newest"] else None
        print(f"  agents {plan}: runs done {st['done']}/{expect}, in progress {st['partial']}, "
              f"rounds {st['rounds']}/{st['total_rounds']}, failed calls {st['failed_calls']}"
              + (f", finished in {fmt(t1 - t0)}" if t1 else f", ETA {fmt(eta)}")
              + (f", last round {stall:.0f} min ago" if stall is not None and not t1 else ""))
    for tag, folder in (("s1", OUT / "runs" / "panel_s1"), ("s23", OUT / "runs" / "panel_s23")):
        t0 = marks.get(f"MONITOR START {tag}")
        t1 = marks.get(f"MONITOR DONE {tag}")
        if "REPAIR MONITOR START" in marks:      # finish pass: full views on s1, Say+Do only on s23
            t0 = marks["REPAIR MONITOR START"]
            t1 = marks.get("REPAIR MONITOR DONE")
        if t0 is None:
            continue
        st = monitor_status(folder)
        if tag == "s23":
            st["expected"] = st["runs"]          # one judge view per run
            st["judge_files"] = len(list(folder.glob("*/monitor/glm__saydo.json")))
        eta = None
        if t1 is None and st["judge_files"] > 0:
            rate = st["judge_files"] / max(1, now - t0)
            eta = (st["expected"] - st["judge_files"]) / rate if rate > 0 else None
        print(f"  monitor {tag}: judge files {st['judge_files']}/{st['expected']} over {st['runs']} runs"
              + (f", finished in {fmt(t1 - t0)}" if t1 else f", ETA {fmt(eta)}"))
    if "ANALYSIS START" in marks:
        print("  analysis: " + ("done" if "PIPELINE DONE" in marks else "running"))
    # worst recent error lines from the step logs
    bad = []
    for p in sorted((OUT / "step_logs").glob("*.txt")):
        for line in p.read_text(encoding="utf8", errors="replace").splitlines():
            if re.search(r"FAIL|STOPPED|Traceback", line):
                bad.append(f"{p.name}: {line[:140]}")
    if bad:
        print("  PROBLEMS:")
        for b in bad[-8:]:
            print("   ", b)


if __name__ == "__main__":
    main()
