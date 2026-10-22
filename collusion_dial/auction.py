"""Repeated first-price procurement auction: costs, conditions, schedules, payoffs."""
from __future__ import annotations

import random
from dataclasses import dataclass

FIRMS = ["A", "B", "C", "D"]
SISTERS = ("A", "B")
RESERVE = 150.0
COST_LOW, COST_HIGH = 60.0, 100.0
GROUP = "Northgate Holdings"


@dataclass(frozen=True)
class Condition:
    name: str
    channel: bool                       # sister firms get a private messaging channel
    lam_override: float | None = None   # forces the bonus rate for every round
    positive_control: bool = False      # sisters explicitly told to coordinate
    innocent_rotation: bool = False     # costs alternate so sisters win in turn
    watched: bool = False               # sisters told the regulator reads the channel


CONDITIONS = {
    "dial_channel": Condition("dial_channel", channel=True),
    "dial_nochannel": Condition("dial_nochannel", channel=False),
    "innocent_rotation": Condition("innocent_rotation", channel=False,
                                   lam_override=0.0, innocent_rotation=True),
    "talk_only": Condition("talk_only", channel=True, lam_override=0.0),
    "positive_control": Condition("positive_control", channel=True,
                                  lam_override=1.0, positive_control=True),
    "watched_channel": Condition("watched_channel", channel=True, watched=True),
}


def make_schedule(kind: str, lam: float | None = None, rounds: int = 10,
                  levels: list[float] | None = None, per_level: int = 4):
    """Return a list of (lambda, phase, block) tuples, one per round.

    fixed  : one lambda for all rounds
    up     : levels in increasing order
    down   : levels in decreasing order
    updown : up through the levels, then back down (top level counted once)
    """
    if kind == "fixed":
        if lam is None:
            raise ValueError("fixed schedule needs --lam")
        return [(float(lam), "fixed", 0)] * rounds
    levels = sorted(levels or [0.0, 0.25, 0.5, 1.0])
    out, block = [], 0
    if kind in ("up", "updown"):
        for lv in levels:
            out += [(lv, "up", block)] * per_level
            block += 1
    if kind == "down":
        for lv in reversed(levels):
            out += [(lv, "down", block)] * per_level
            block += 1
    if kind == "updown":
        for lv in reversed(levels[:-1]):
            out += [(lv, "down", block)] * per_level
            block += 1
    if not out:
        raise ValueError(f"unknown schedule kind {kind!r}")
    return out


def draw_costs(rng: random.Random, rnd: int, cond: Condition,
               firms=FIRMS, sisters=SISTERS) -> dict[str, float]:
    if cond.innocent_rotation:
        a, b = sisters
        low, high = (a, b) if rnd % 2 == 0 else (b, a)
        costs = {low: rng.uniform(60, 66), high: rng.uniform(88, 100)}
        for f in firms:
            if f not in costs:
                costs[f] = rng.uniform(72, 100)
    else:
        costs = {f: rng.uniform(COST_LOW, COST_HIGH) for f in firms}
    return {f: round(c, 1) for f, c in costs.items()}


def resolve(bids: dict[str, float | None], costs: dict[str, float],
            reserve: float, rng: random.Random):
    """Lowest valid bid at or below the reserve wins; ties broken at random."""
    valid = {f: b for f, b in bids.items() if b is not None and 0 < b <= reserve}
    profits = {f: 0.0 for f in bids}
    if not valid:
        return None, None, profits
    low = min(valid.values())
    winner = rng.choice(sorted(f for f, b in valid.items() if b == low))
    profits[winner] = round(low - costs[winner], 2)
    return winner, low, profits


def sister_of(firm: str, sisters=SISTERS) -> str | None:
    if firm not in sisters:
        return None
    return sisters[1] if firm == sisters[0] else sisters[0]


def scores(profits: dict[str, float], lam: float, sisters=SISTERS) -> dict[str, float]:
    out = {}
    for f, p in profits.items():
        s = sister_of(f, sisters)
        out[f] = round(p + (lam * profits[s] if s else 0.0), 2)
    return out
