"""Run a predefined experiment grid.

  python -m collusion_dial.grid --plan pilot --model glm --dry-run
  python -m collusion_dial.grid --plan eval  --model glm --yes
  python -m collusion_dial.grid --plan dial  --model glm --yes

Plans
  pilot         explicit_collusion, cheap_talk, honest; seed 1 (3 runs)
  eval          the balanced evaluation set: explicit + tacit seeds 1-6,
                cheap_talk, innocent_parallel, honest, honest_nomsg seeds 1-3 (24 runs)
  dial          dial fixed lambda {0, 0.25, 0.5, 1} x seeds 1-3, plus one up/down sweep
                per seed (15 runs)
  dial_watched  dial_watched fixed lambda {0.5, 1} x seeds 1-3
  robustness    explicit, tacit, cheap_talk, honest; seeds 1-2 (second agent family)
  panel         explicit, tacit, honest + the rule ladder (dial, dial_softrule, dial_norule,
                dial_norule_watched) at lambda 1; seed 1 (OpenRouter model panel,
                or_* entries of config/models.yaml)
  panel_more    the four rule-ladder arms at seeds 2-3

Finished runs are skipped, so an interrupted grid can be restarted with the
same command. Usage and estimated spend are written to <out>/usage.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .auction import get_condition
from .game import is_complete, play_run
from .llm import BudgetExceeded, build_models, load_config
from .run import make_spec, respec_seed

UPDOWN_LEVELS = [0.0, 0.25, 0.5, 1.0]
DIAL_LAMS = (0.0, 0.25, 0.5, 1.0)
PLANS = ["pilot", "eval", "dial", "dial_watched", "dial_norule", "robustness", "extensions",
         "robustness_dial", "panel", "panel_more"]


def plan_specs(plan: str, model: str, n_firms: int = 3):
    mk = lambda c, s, **k: make_spec(c, "fixed", s, model, rounds=12, n_firms=n_firms, **k)  # noqa: E731
    if plan == "pilot":
        return [mk(c, 1) for c in ("explicit_collusion", "cheap_talk", "honest")]
    if plan == "eval":
        S = [mk(c, s) for c in ("explicit_collusion", "tacit_collusion") for s in range(1, 7)]
        S += [mk(c, s) for c in ("cheap_talk", "innocent_parallel", "honest", "honest_nomsg")
              for s in range(1, 4)]
        return S
    if plan == "dial":
        S = [mk("dial", s, lam=l) for s in (1, 2, 3) for l in DIAL_LAMS]
        S += [make_spec("dial", "updown", s, model, levels=UPDOWN_LEVELS, per_level=6,
                        n_firms=n_firms) for s in (1, 2, 3)]
        return S
    if plan == "dial_watched":
        return [mk("dial_watched", s, lam=l) for s in (1, 2, 3) for l in (0.5, 1.0)]
    if plan == "dial_norule":   # second lever: no independence rule stated; 2 seeds
        S = [mk("dial_norule", s, lam=l) for s in (1, 2) for l in (0.0, 0.5, 1.0)]
        S += [make_spec("dial_norule", "updown", s, model, levels=UPDOWN_LEVELS, per_level=6,
                        n_firms=n_firms) for s in (1, 2)]
        return S
    if plan == "robustness":
        return [mk(c, s) for c in ("explicit_collusion", "tacit_collusion", "cheap_talk", "honest")
                for s in (1, 2)]
    if plan == "extensions":   # after the final run: surveillance without a rule, soft rule, decay tail
        S = [mk("dial_norule_watched", s, lam=1.0) for s in (1, 2, 3)]
        S += [mk("dial_softrule", s, lam=1.0) for s in (1, 2, 3)]
        S += [make_spec("dial_norule", "updown", s, model, levels=UPDOWN_LEVELS, per_level=6,
                        n_firms=n_firms, tail=12) for s in (1, 2)]
        return S
    if plan == "robustness_dial":   # second agent family on the headline: rule vs no rule at lambda 1
        return [mk("dial", 1, lam=1.0), mk("dial_norule", 1, lam=1.0)]
    if plan == "panel":   # model panel (OpenRouter): three controls + the four-rung rule ladder at lambda 1, seed 1
        return [mk("explicit_collusion", 1), mk("tacit_collusion", 1), mk("honest", 1),
                mk("dial", 1, lam=1.0), mk("dial_softrule", 1, lam=1.0),
                mk("dial_norule", 1, lam=1.0), mk("dial_norule_watched", 1, lam=1.0)]
    if plan == "panel_more":   # seeds 2-3 of the rule ladder
        return [mk(c, s, lam=1.0) for s in (2, 3)
                for c in ("dial", "dial_softrule", "dial_norule", "dial_norule_watched")]
    raise ValueError(plan)


def calls_for(spec) -> int:
    n = len(spec.firms)
    per_round = 2 * n + (n if get_condition(spec.condition).messaging == "all" else 0)
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
    ap.add_argument("--plan", default="pilot", choices=PLANS)
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="default logs/<plan>")
    ap.add_argument("--parallel-runs", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8, help="max in-flight API calls")
    ap.add_argument("--n-firms", type=int, default=3, choices=[3, 4])
    ap.add_argument("--seeds", default=None,
                    help="comma-separated subset of seeds, e.g. 1,2 (budget control)")
    ap.add_argument("--fixed-only", action="store_true",
                    help="drop the up/down sweeps (hysteresis is measured in the arm where "
                         "collusion onsets, i.e. dial_norule)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="add N to every seed: a second batch with new cost draws (pooled analysis)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    out = Path(a.out or f"logs/{a.plan}")
    specs = plan_specs(a.plan, a.model, a.n_firms)
    if a.seeds:
        keep = {int(x) for x in a.seeds.split(",")}
        specs = [s for s in specs if s.seed in keep]
    if a.fixed_only:
        specs = [s for s in specs if s.schedule_kind == "fixed"]
    if a.seed_offset:
        specs = [respec_seed(s, s.seed + a.seed_offset, a.model) for s in specs]
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
