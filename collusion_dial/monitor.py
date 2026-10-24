"""Run monitors over saved logs: claim extraction, antitrust screens,
cross-agent consistency and the LLM judge views.

  python -m collusion_dial.monitor --logs logs/eval --judge-model qwen
  python -m collusion_dial.monitor --logs logs/dial --judge-model qwen
  python -m collusion_dial.monitor --logs logs/dial --judge-model qwen --claims-only --include-sweeps
  python -m collusion_dial.monitor --logs logs/eval --judge-model qwen --views saydo,structured --stake 0.5

Agents are never rerun: monitors read the saved transcripts (post-hoc scoring,
as in TraceGuard). Results go to <run_dir>/monitor/:
  screens.json                 bid screens per pair (no LLM)
  claims__<model>.json         claims extracted from messages and public statements
  claims_notes__<model>.json   claims extracted from private notes (--notes; lying analysis only)
  consistency.json             claim-vs-outcome scores per pair
  <model>__<view>[__stakeX].json   judge output per view
Existing results are kept. Sweeps are skipped unless --include-sweeps.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .llm import BudgetExceeded, build_models, load_config
from .metrics.collusion import load_run
from .monitors.claims import extract_claims, score_claims
from .monitors.judge import judge_run, mock_signals_for
from .monitors.merit import SIGMAS, cost_estimates, merit_round_table
from .monitors.relabel import perm_for
from .monitors.screens import compute_screens
from .monitors.views import VIEWS


def run_dirs(logs: Path, include_sweeps: bool):
    for meta_path in sorted(logs.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        if meta["schedule_kind"] != "fixed" and not include_sweeps:
            continue
        yield meta_path.parent


async def ensure_claims(run_dir, meta, rounds, perm, model, sem, source):
    name = "claims" if source == "messages" else "claims_notes"
    path = run_dir / "monitor" / f"{name}__{model.key}.json"
    if path.exists():
        return json.loads(path.read_text())["items"]
    async with sem:
        res = await extract_claims(model, meta, rounds, perm, source)
    path.write_text(json.dumps({**res, "perm": perm}, indent=2))
    print(f"  {run_dir.name}  {path.name}  items={res['n_items']} ok={res['ok']}", flush=True)
    return res["items"]


async def score_run(run_dir, judge, views, stake, sem, claims_only=False, notes=False,
                    extractor=None, merit_sigmas=SIGMAS):
    extractor = extractor or judge
    meta, rounds = load_run(run_dir)
    firms = meta["firms"]
    mdir = run_dir / "monitor"
    mdir.mkdir(exist_ok=True)
    (mdir / "screens.json").write_text(
        json.dumps(compute_screens(rounds, firms, meta["reserve"]), indent=2))
    for sigma in merit_sigmas:   # privileged auditor tables, no LLM; scored in evaluate
        merit_round_table(meta, rounds, cost_estimates(meta, rounds, sigma)).to_csv(
            mdir / f"merit_table__s{sigma:g}.csv", index=False)
    perm = perm_for(meta["run_id"], firms)
    claims = await ensure_claims(run_dir, meta, rounds, perm, extractor, sem, "messages")
    if notes:
        await ensure_claims(run_dir, meta, rounds, perm, extractor, sem, "notes")
    scored = score_claims(claims, rounds, firms)
    (mdir / "consistency.json").write_text(
        json.dumps({"model": extractor.key, "perm": perm, **scored}, indent=2))
    if claims_only:
        return
    signals = mock_signals_for(meta, rounds, scored, perm) if judge.is_mock else None

    async def one(view):
        name = f"{judge.key}__{view}" + (f"__stake{stake:g}" if stake else "") + ".json"
        path = mdir / name
        if path.exists():
            return
        async with sem:
            res = await judge_run(judge, meta, rounds, view, perm, stake=stake, claims=claims,
                                  scored=scored, mock_signals=signals)
        path.write_text(json.dumps(res, indent=2))
        print(f"  {run_dir.name}  {name}  ok={res['ok']}", flush=True)

    await asyncio.gather(*[one(v) for v in views])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    ap.add_argument("--judge-model", default="glm")
    ap.add_argument("--extractor-model", default=None,
                    help="model for claim extraction (default: the judge model). A literal "
                         "parsing task; a reasoning judge can take minutes per batch.")
    ap.add_argument("--views", default=",".join(VIEWS),
                    help="comma-separated; add do_merit for the optional privileged LLM view")
    ap.add_argument("--merit-sigmas", default=",".join(f"{s:g}" for s in SIGMAS),
                    help="noise levels of the auditor's cost estimates (no LLM calls)")
    ap.add_argument("--stake", type=float, default=None, help="judge bonus share (extension)")
    ap.add_argument("--include-sweeps", action="store_true")
    ap.add_argument("--claims-only", action="store_true",
                    help="extract claims and compute screens/consistency; no judge calls")
    ap.add_argument("--notes", action="store_true",
                    help="also extract claims from private notes (lying typology only; the regex "
                         "fallback is used otherwise). Costly with a reasoning judge.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()

    cfg = load_config(a.config)
    ext_key = a.extractor_model or a.judge_model
    models, tracker = build_models(cfg, [a.judge_model, ext_key], a.concurrency)
    judge, extractor = models[a.judge_model], models[ext_key]
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    dirs = list(run_dirs(Path(a.logs), a.include_sweeps))
    print(f"{len(dirs)} runs, judge={a.judge_model}, extractor={ext_key}, "
          f"{'claims only' if a.claims_only else 'views=' + ','.join(views)}")

    sigmas = [float(x) for x in a.merit_sigmas.split(",") if x.strip()]

    async def go():
        sem = asyncio.Semaphore(a.concurrency)
        await asyncio.gather(*[score_run(d, judge, views, a.stake, sem, a.claims_only,
                                         a.notes, extractor, sigmas) for d in dirs])

    try:
        asyncio.run(go())
    except BudgetExceeded as e:
        print(f"STOPPED: {e}")
    print(json.dumps(tracker.summary(), indent=2))


if __name__ == "__main__":
    main()
