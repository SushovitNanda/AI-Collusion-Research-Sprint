"""Check an endpoint works before spending credits on a grid.

  python -m collusion_dial.smoke --model glm
  python -m collusion_dial.smoke --model qwen

Sends one tiny JSON-schema request and one real auction-stage request, then
prints the parsed output, whether native reasoning came back, and token usage.
"""
from __future__ import annotations

import argparse
import asyncio
import json

from . import prompts as P
from .auction import CONDITIONS, FIRMS, SISTERS
from .llm import build_models, load_config

TINY = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                         "city": {"type": "string"}},
        "required": ["name", "age", "city"], "additionalProperties": False}


async def main_async(key, config):
    cfg = load_config(config)
    models, tracker = build_models(cfg, [key], 2)
    m = models[key]
    r1 = await m.complete([{"role": "user", "content": "Extract the name, age, and city: Ada, 36, London."}],
                          "person_info", TINY, max_tokens=200)
    print("tiny  :", r1["data"], "| error:", r1["error"])
    cond = CONDITIONS["dial_channel"]
    ctx = dict(firm="A", sister="B", round=0, cost=78.4, lam=0.5, prev_lam=0.0,
               policy_changed=False, total_score=0.0, history=[], window=8, channel=True,
               chat_history=[], last_notes="")
    r2 = await m.complete([{"role": "system", "content": P.firm_system_prompt("A", FIRMS, SISTERS, cond, 150)},
                           {"role": "user", "content": P.think_prompt(ctx)}],
                          "plan", P.THINK_SCHEMA, meta=dict(run_id="smoke", firm="A", sister="B",
                          is_sister=True, sisters=SISTERS, round=0, lam=0.5, cost=78.4,
                          reserve=150, condition="dial_channel", positive_control=False),
                          max_tokens=400)
    print("plan  :", r2["data"], "| error:", r2["error"])
    print("native reasoning returned:", bool(r2["reasoning"]))
    print(json.dumps(tracker.summary(), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="glm")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    asyncio.run(main_async(a.model, a.config))


if __name__ == "__main__":
    main()
