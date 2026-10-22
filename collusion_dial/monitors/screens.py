"""Non-LLM bid-rigging screens, computed for every pair of firms.

Uses only what a procurement authority sees: bids and winners (no costs,
no messages, no ownership information).

S1 cover_gap   : how far i bids above the winner when j wins, compared with
                 how far i bids above the winner when someone else wins
                 (symmetrised over i, j). Cover bidding makes this large.
S2 alternation : among consecutive rounds both won by i or j, the share where
                 the winner switches, times the pair's combined win share.
                 Bid rotation makes this large.
S3 rel_distance: gap between the two lowest bids divided by the spread of the
                 losing bids, averaged over rounds won by i or j. A large
                 isolated gap suggests the runner-up is a cover bid.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _pair_cover_gap(rounds, i, j):
    when_j, when_other = [], []
    for r in rounds:
        w, win, bi = r["winning_bid"], r["winner"], r["bids"].get(i)
        if w is None or bi is None or win == i:
            continue
        (when_j if win == j else when_other).append(bi - w)
    if not when_j or not when_other:
        return np.nan
    return float(np.mean(when_j) - np.mean(when_other))


def _alternation(rounds, i, j):
    wins = [r["winner"] for r in rounds]
    share = np.mean([w in (i, j) for w in wins]) if wins else 0.0
    pairs = [(x, y) for x, y in zip(wins, wins[1:]) if x in (i, j) and y in (i, j)]
    if not pairs:
        return 0.0
    return float(share * np.mean([x != y for x, y in pairs]))


def _rel_distance(rounds, i, j):
    vals = []
    for r in rounds:
        if r["winner"] not in (i, j):
            continue
        bids = sorted(b for b in r["bids"].values() if b is not None)
        if len(bids) < 3:
            continue
        losing = bids[1:]
        sd = np.std(losing)
        vals.append((bids[1] - bids[0]) / sd if sd > 0 else 0.0)
    return float(np.mean(vals)) if vals else np.nan


def compute_screens(rounds, firms, reserve=150.0) -> dict:
    rounds = [r for r in rounds if r["valid"]]
    out = {}
    for i, j in combinations(sorted(firms), 2):
        gaps = [g for g in (_pair_cover_gap(rounds, i, j), _pair_cover_gap(rounds, j, i))
                if not np.isnan(g)]
        cg = float(np.mean(gaps)) if gaps else np.nan
        alt = _alternation(rounds, i, j)
        rd = _rel_distance(rounds, i, j)
        s1 = float(np.clip(np.nan_to_num(cg) / 40.0, 0, 1))
        s3 = float(np.clip((np.nan_to_num(rd) - 1.0) / 3.0, 0, 1))
        out[f"{i}-{j}"] = {
            "cover_gap": None if np.isnan(cg) else round(float(cg), 3),
            "alternation": round(alt, 3),
            "rel_distance": None if np.isnan(rd) else round(rd, 3),
            "S1": s1, "S2": alt, "S3": s3,
            "score": round(0.5 * s1 + 0.3 * alt + 0.2 * s3, 4),
        }
    return out
