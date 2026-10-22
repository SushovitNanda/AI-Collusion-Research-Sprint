"""Play a single run.

Examples
  python -m collusion_dial.run --condition dial_channel --schedule fixed --lam 0.5 --seed 1
  python -m collusion_dial.run --condition dial_channel --schedule updown \
      --levels 0,0.25,0.5,1 --per-level 4 --seed 1
  python -m collusion_dial.run --model mock ...      # offline test, no credits used
"""
from __future__ import annotations

import argparse
import asyncio
import json

from .auction import CONDITIONS, FIRMS, make_schedule
from .game import RunSpec, play_run
from .llm import build_models, load_config


def run_id_for(condition, kind, lam, seed, model, levels=None, n_firms=3):
    lam_part = f"lam{lam:g}" if kind == "fixed" else f"{kind}{'-'.join(f'{x:g}' for x in levels)}"
    n_part = f"__n{n_firms}"
    return f"{model}__{condition}__{lam_part}{n_part}__s{seed}"


def make_spec(condition, kind, seed, model_key, lam=None, rounds=12, levels=None, per_level=6,
              n_firms=3):
    cond = CONDITIONS[condition]
    if cond.lam_override is not None and kind == "fixed":
        lam = cond.lam_override
    sched = make_schedule(kind, lam=lam, rounds=rounds, levels=levels, per_level=per_level)
    if not 3 <= n_firms <= len(FIRMS):
        raise ValueError("n_firms must be 3 or 4")
    return RunSpec(run_id=run_id_for(condition, kind, lam, seed, model_key, levels or [], n_firms),
                   condition=condition, schedule=sched, seed=seed, schedule_kind=kind,
                   lam_label=lam, agent_model=model_key, firms=list(FIRMS[:n_firms]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="dial_channel", choices=sorted(CONDITIONS))
    ap.add_argument("--schedule", default="fixed", choices=["fixed", "up", "down", "updown"])
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--levels", default="0,0.25,0.5,1")
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="glm", help="key in config/models.yaml")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="logs/adhoc")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-firms", type=int, default=3, choices=[3, 4])
    a = ap.parse_args()

    levels = [float(x) for x in a.levels.split(",")]
    spec = make_spec(a.condition, a.schedule, a.seed, a.model, lam=a.lam, rounds=a.rounds,
                     levels=levels, per_level=a.per_level, n_firms=a.n_firms)
    cfg = load_config(a.config)
    models, tracker = build_models(cfg, [a.model], a.concurrency)
    path = asyncio.run(play_run(spec, models[a.model], a.out))
    print(f"wrote {path}")
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    main()
