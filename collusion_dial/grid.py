"""Run a predefined experiment grid.

  python -m collusion_dial.grid --plan pilot --model glm --dry-run
  python -m collusion_dial.grid --plan solo  --model glm --yes

Finished runs are skipped, so an interrupted grid can be restarted with the
same command. Usage and estimated spend are written to <out>/usage.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .auction import CONDITIONS
from .game import is_complete, play_run
from .llm import BudgetExceeded, build_models, load_config
from .run import make_spec

UPDOWN_LEVELS = [0.0, 0.25, 0.5, 1.0]


def plan_specs(plan: str, model: str, n_firms: int = 3):
    S = []
    mk = lambda *a, **k: make_spec(*a, n_firms=n_firms, **k)  # noqa: E731
    if plan == "pilot":
        for seed in (1, 2):
            S.append(mk("positive_control", "fixed", seed, model, rounds=5))
            S.append(mk("dial_channel", "fixed", seed, model, lam=0.0, rounds=5))
        return S

    if plan == "solo":
        seeds, lams, conds = (1, 2, 3), (0.0, 0.25, 0.5, 1.0), ("dial_channel",)
        controls = ("positive_control", "innocent_rotation")
    elif plan == "full":
        seeds, lams = (1, 2, 3, 4), (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
        conds = ("dial_channel", "dial_nochannel")
        controls = ("positive_control", "innocent_rotation", "talk_only")
    elif plan == "watched":   # optional extension: regulator reads the channel
        return [mk("watched_channel", "fixed", s, model, lam=l, rounds=12)
                for s in (1, 2, 3) for l in (0.5, 1.0)]
    elif plan == "robustness":  # second model family on the extremes of the dial
        return [mk("dial_channel", "fixed", s, model, lam=l, rounds=12)
                for s in (1, 2) for l in (0.0, 1.0)] + \
               [mk("positive_control", "fixed", s, model, rounds=12) for s in (1, 2)]
    else:
        raise ValueError(plan)

    levels = UPDOWN_LEVELS if plan == "solo" else [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    for seed in seeds:
        for c in conds:
            for lam in lams:
                S.append(mk(c, "fixed", seed, model, lam=lam, rounds=12))
            S.append(mk(c, "updown", seed, model, levels=levels, per_level=6))
        for c in controls:
            S.append(mk(c, "fixed", seed, model, rounds=12))
    return S


def calls_for(spec) -> int:
    per_round = len(spec.firms) * 2 + (2 if CONDITIONS[spec.condition].channel else 0)
    return per_round * len(spec.schedule)


async def run_all(specs, model, out, parallel_runs):
    sem = asyncio.Semaphore(parallel_runs)

    async def one(s):
        async with sem:
            try:
                p = await play_run(s, model, out)
                print(f"done  {p.name}", flush=True)
            except BudgetExceeded:
                raise
            except Exception as e:
                print(f"FAIL  {s.run_id}: {type(e).__name__}: {e}", flush=True)

    await asyncio.gather(*[one(s) for s in specs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="pilot",
                    choices=["pilot", "solo", "full", "watched", "robustness"])
    ap.add_argument("--model", default="glm")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default logs/<plan>")
    ap.add_argument("--parallel-runs", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8, help="max in-flight API calls")
    ap.add_argument("--n-firms", type=int, default=3, choices=[3, 4],
                    help="3 firms makes the 2-firm ring compete head-to-head more often")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    out = Path(a.out or f"logs/{a.plan}")
    specs = plan_specs(a.plan, a.model, a.n_firms)
    todo = [s for s in specs if not is_complete(out / s.run_id)]
    n_calls = sum(calls_for(s) for s in todo)
    print(f"plan={a.plan} model={a.model} runs={len(specs)} remaining={len(todo)} "
          f"~agent calls={n_calls:,} (plus retries)")
    for s in todo:
        print(f"  {s.run_id}  rounds={len(s.schedule)}  calls={calls_for(s)}")
    if a.dry_run or not todo:
        return
    if not a.yes and input("Proceed? [y/N] ").strip().lower() != "y":
        return

    cfg = load_config(a.config)
    models, tracker = build_models(cfg, [a.model], a.concurrency)
    out.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(run_all(todo, models[a.model], out, a.parallel_runs))
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
    finally:
        usage_path = out / "usage.json"
        prev = json.loads(usage_path.read_text()) if usage_path.exists() else []
        prev.append({"plan": a.plan, "model": a.model, "usage": tracker.summary()})
        usage_path.write_text(json.dumps(prev, indent=2))
        print(json.dumps(tracker.summary(), indent=2))
        s = tracker.summary().get(a.model)
        if s and s["calls"]:
            print(f"avg tokens/call: in={s['input_tokens'] / s['calls']:.0f} "
                  f"out={s['output_tokens'] / s['calls']:.0f}; "
                  f"parse failures={s['parse_failures']}, api errors={s['api_errors']}")


if __name__ == "__main__":
    main()
