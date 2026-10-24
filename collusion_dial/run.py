"""Play a single run.

Examples
  python -m collusion_dial.run --condition dial --schedule fixed --lam 0.5 --seed 1
  python -m collusion_dial.run --condition explicit_collusion --seed 1
  python -m collusion_dial.run --condition dial --schedule updown \
      --levels 0,0.25,0.5,1 --per-level 6 --seed 1
  python -m collusion_dial.run --model mock ...      # offline test, no credits used
"""
from __future__ import annotations

import argparse
import asyncio
import json

from .auction import ALIASES, CONDITIONS, FIRMS, get_condition, make_schedule
from .game import RunSpec, play_run
from .llm import build_models, load_config


def run_id_for(condition, kind, lam, seed, model, levels=None, n_firms=3, tail=0):
    lam_part = f"lam{lam:g}" if kind == "fixed" else f"{kind}{'-'.join(f'{x:g}' for x in levels)}"
    if tail:
        lam_part += f"-tail{tail}"
    return f"{model}__{condition}__{lam_part}__n{n_firms}__s{seed}"


def make_spec(condition, kind, seed, model_key, lam=None, rounds=12, levels=None, per_level=6,
              n_firms=3, tail=0):
    cond = get_condition(condition)
    if cond.lam_override is not None and kind == "fixed":
        lam = cond.lam_override
    if kind == "fixed" and lam is None:
        raise ValueError(f"condition {cond.name} needs --lam")
    sched = make_schedule(kind, lam=lam, rounds=rounds, levels=levels, per_level=per_level, tail=tail)
    if not 3 <= n_firms <= len(FIRMS):
        raise ValueError("n_firms must be 3 or 4")
    return RunSpec(run_id=run_id_for(cond.name, kind, lam, seed, model_key, levels or [], n_firms, tail),
                   condition=cond.name, schedule=sched, seed=seed, schedule_kind=kind,
                   lam_label=lam, agent_model=model_key, firms=list(FIRMS[:n_firms]))


def respec_seed(spec, seed, model_key):
    """Same run with another seed (new cost draws); keeps the schedule."""
    levels = sorted({l for l, _, _ in spec.schedule})
    tail = sum(1 for _, p, _ in spec.schedule if p == "tail")
    spec.seed = seed
    spec.run_id = run_id_for(spec.condition, spec.schedule_kind, spec.lam_label, seed, model_key,
                             levels, len(spec.firms), tail)
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="dial", choices=sorted(CONDITIONS) + sorted(ALIASES))
    ap.add_argument("--schedule", default="fixed", choices=["fixed", "up", "down", "updown"])
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--levels", default="0,0.25,0.5,1")
    ap.add_argument("--per-level", type=int, default=6)
    ap.add_argument("--tail", type=int, default=0, help="extra rounds at the final level after a sweep")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="qwen", help="key in config/models.yaml")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="logs/adhoc")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--n-firms", type=int, default=3, choices=[3, 4])
    a = ap.parse_args()

    levels = [float(x) for x in a.levels.split(",")]
    spec = make_spec(a.condition, a.schedule, a.seed, a.model, lam=a.lam, rounds=a.rounds,
                     levels=levels, per_level=a.per_level, n_firms=a.n_firms, tail=a.tail)
    cfg = load_config(a.config)
    models, tracker = build_models(cfg, [a.model], a.concurrency)
    path = asyncio.run(play_run(spec, models[a.model], a.out))
    print(f"wrote {path}")
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    main()
