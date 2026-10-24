"""Plays one run of the repeated auction and writes it to disk.

Output: <out_dir>/<run_id>/meta.json and rounds.jsonl (one line per round,
flushed as it goes, so a crashed run keeps its finished rounds).

Messaging (v0.2): when the condition's messaging mode is 'all', every firm,
in an order that rotates each round, may send one private message to any one
other firm. A firm sees only messages it sent or received.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import prompts as P
from .auction import FIRMS, RESERVE, SISTERS, draw_costs, get_condition, resolve, scores, sister_of
from .llm import ChatModel

MAX_TOKENS = {"plan": 400, "channel": 200, "act": 300}
VERSION = "0.2.0"


@dataclass
class RunSpec:
    run_id: str
    condition: str
    schedule: list            # [(lam, phase, block), ...]
    seed: int
    schedule_kind: str = "fixed"
    lam_label: float | None = None
    agent_model: str = "glm"
    firms: list = field(default_factory=lambda: list(FIRMS[:3]))
    sisters: tuple = SISTERS
    reserve: float = RESERVE
    history_window: int = 8


def _bid(x):
    try:
        v = round(float(x), 2)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def store_message(round_chat: list, rnd: int, sender: str, data, firms) -> dict | None:
    """Validate one channel reply; append and return the stored message, or None.
    Empty text, to == NONE, unknown recipients and messages to self are dropped."""
    d = data or {}
    to = str(d.get("to") or "NONE").strip().upper()
    text = str(d.get("message") or "").strip()
    if to == "NONE" or not text or to == sender or to not in firms:
        return None
    m = {"round": rnd, "from": sender, "to": to, "text": text}
    round_chat.append(m)
    return m


def involving(messages, firm):
    return [m for m in messages if firm in (m["from"], m["to"])]


def is_complete(run_dir: Path) -> bool:
    meta, rounds = run_dir / "meta.json", run_dir / "rounds.jsonl"
    if not (meta.exists() and rounds.exists()):
        return False
    n = len(json.loads(meta.read_text())["schedule"])
    return sum(1 for _ in rounds.open()) >= n


async def play_run(spec: RunSpec, model: ChatModel, out_dir: str | Path) -> Path:
    cond = get_condition(spec.condition)
    run_dir = Path(out_dir) / spec.run_id
    if is_complete(run_dir):
        return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {**asdict(spec), "condition": cond.name, "version": VERSION,
            "model_config_key": model.key, "started": time.time(),
            "condition_flags": asdict(cond)}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    cost_rng = random.Random(spec.seed * 1000 + 7)    # same costs for the same seed
    tie_rng = random.Random(spec.seed * 1000 + 13)
    firms = list(spec.firms)
    systems = {f: P.firm_system_prompt(f, firms, spec.sisters, cond, spec.reserve) for f in firms}
    # Freeze proof: the report can show every run of a condition used identical prompts.
    meta["prompt_sha256"] = hashlib.sha256("\n".join(systems[f] for f in firms).encode()).hexdigest()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    history, chat_history = [], []
    totals = {f: 0.0 for f in firms}
    last_notes = {f: "" for f in firms}
    prev_lam = None

    with (run_dir / "rounds.jsonl").open("w") as fh:
        for rnd, (lam, phase, block) in enumerate(spec.schedule):
            lam_eff = cond.lam_override if cond.lam_override is not None else lam
            costs = draw_costs(cost_rng, rnd, cond, firms, spec.sisters)
            changed = prev_lam is not None and lam_eff != prev_lam
            calls = []

            def ctx(f):
                return dict(firm=f, sister=sister_of(f, spec.sisters), round=rnd,
                            cost=costs[f], lam=lam_eff, prev_lam=prev_lam or 0.0,
                            policy_changed=changed, total_score=totals[f],
                            history=history, window=spec.history_window,
                            messaging=cond.messaging,
                            chat_history=involving(chat_history, f),
                            last_notes=last_notes[f])

            def meta_for(f, **kw):
                # meta is read only by the offline mock (all costs let it script the
                # explicit ring's lower-cost-wins style); real models never see it
                return dict(run_id=spec.run_id, firm=f, sister=sister_of(f, spec.sisters),
                            is_sister=f in spec.sisters, sisters=spec.sisters, firms=firms,
                            round=rnd, lam=lam_eff, cost=costs[f], costs=costs,
                            reserve=spec.reserve, condition=cond.name, label=cond.label,
                            messaging=cond.messaging,
                            instructed=cond.group_coordination_permitted, **kw)

            async def call(f, stage, user, schema, **kw):
                res = await model.complete(
                    [{"role": "system", "content": systems[f]},
                     {"role": "user", "content": user}],
                    stage, schema, meta=meta_for(f, **kw), max_tokens=MAX_TOKENS[stage])
                calls.append({"firm": f, "stage": stage, "ok": res["data"] is not None,
                              "error": res["error"], **res["usage"],
                              "native_reasoning": res["reasoning"]})
                return res["data"]

            # Stage 1: private plans
            plans_raw = await asyncio.gather(
                *[call(f, "plan", P.think_prompt(ctx(f)), P.THINK_SCHEMA) for f in firms])
            plans = {f: (p or {"notes": "", "intended_bid": None})
                     for f, p in zip(firms, plans_raw)}

            # Stage 2: private messages, every firm, rotating order; each sees its inbox
            round_chat = []
            if cond.messaging == "all":
                k = rnd % len(firms)
                order = firms[k:] + firms[:k]
                for f in order:
                    inbox = [m for m in round_chat if m["to"] == f]
                    d = await call(f, "channel", P.channel_prompt(ctx(f), plans[f], inbox, firms),
                                   P.channel_schema(f, firms), inbox=inbox,
                                   intended_bid=plans[f].get("intended_bid"))
                    store_message(round_chat, rnd, f, d, firms)

            # Stage 3: sealed bids, public statements, declarations
            acts_raw = await asyncio.gather(*[
                call(f, "act", P.act_prompt(ctx(f), plans[f], involving(round_chat, f)),
                     P.ACT_SCHEMA, intended_bid=plans[f].get("intended_bid"),
                     inbox=involving(round_chat, f)) for f in firms])
            acts = {}
            for f, a in zip(firms, acts_raw):
                a = a or {}
                acts[f] = {"bid": _bid(a.get("bid")),
                           "public_statement": a.get("public_statement", ""),
                           "declaration": a.get("declaration")}

            bids = {f: acts[f]["bid"] for f in firms}
            winner, wbid, profits = resolve(bids, costs, spec.reserve, tie_rng)
            sc = scores(profits, lam_eff, spec.sisters)
            for f in firms:
                totals[f] += sc[f]
                last_notes[f] = plans[f].get("notes", "")

            rec = dict(round=rnd, lam=lam_eff, phase=phase, block=block, costs=costs,
                       plans=plans, chat=round_chat, acts=acts, bids=bids, winner=winner,
                       winning_bid=wbid, profits=profits, scores=sc,
                       valid=all(c["ok"] for c in calls if c["stage"] == "act"),
                       calls=calls)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            history.append(rec)
            chat_history.extend(round_chat)
            prev_lam = lam_eff

    meta["finished"] = time.time()
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return run_dir
