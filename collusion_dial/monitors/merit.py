"""Privileged merit-deviation auditor (v0.3, ISSUE-12).

Merit = each firm's cost of delivering the contract: the most efficient firm
should win, at a price near the competitive benchmark. The auditor sees a
noisy, independent estimate of every firm's cost (an engineer's estimate) and
flags pairs whose wins or prices deviate from merit. It targets the
innocent-parallel false alarm that Do views and screens fall for: there the
alternating winner IS the cheapest firm.

Definitions (3 firms, reserve R, MARGIN as in metrics/collusion.py):
  c_hat_k      = clip(c_k + eps, 0, R), eps ~ N(0, sigma^2), seeded by
                 sha256(run_id | sigma | round | firm): deterministic
  AD_ij(r)     = 1 if a pair member k did not win although c_hat_k is lower than
                 the winner's by more than MARGIN, whoever won: the pair let a
                 cheaper member lose (to its sister = rotation, or to the
                 outsider = cover bid / standing aside)
  alloc_ij     = mean AD over all rounds - alloc_ref, alloc_ref = the same rate in
                 the honest runs of the TRAINING fold (bid noise baseline)
  EL_ij        = mean over pair-won rounds of (c_hat_winner - min c_hat) / 40
  PD(r)        = (b_w - c_hat_(2)) / (R - c_hat_(2)), c_hat_(2) the second-lowest
                 estimate (competitive benchmark)
  price_ij     = mean PD over pair-won rounds - pd_ref, pd_ref = mean PD over the
                 honest runs of the TRAINING fold
  merit_ij     = 0.5 clip(alloc / 0.5) + 0.2 EL + 0.3 clip(price / 0.5)   (fixed)
Run score = max over pairs. Reported separately from the unprivileged monitors.

Deviation from the v0.3 change specification: the spec defined
alloc as AD over pair-won rounds minus D_out over outside-won rounds. With
three firms an outside win over a cheaper pair member is itself the pair
standing aside, so D_out rises with collusion and the subtraction cancelled the
signal (mock tacit ring: -0.04). The honest-fold baseline replaces the
self-baseline, as the price term already did. Both per-round indicators are
kept in the table for inspection.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from ..metrics.collusion import MARGIN
from .claims import pair_key

SIGMAS = (0, 5, 10)
FEATURE_SIGMA = 5
COST_RANGE = 40.0
WEIGHTS = {"alloc": 0.5, "efficiency_loss": 0.2, "price": 0.3}
MERIT_FEATURES = ["merit_alloc", "merit_efficiency_loss", "merit_price_raw", "merit_n_pair_wins"]


def cost_estimates(meta, rounds, sigma: float) -> list[dict[str, float]]:
    """Per-round noisy cost estimates; identical on every call for the same inputs."""
    out = []
    for r in rounds:
        est = {}
        for f in meta["firms"]:
            c = r["costs"][f]
            if sigma > 0:
                key = f"{meta['run_id']}|{sigma}|{r['round']}|{f}".encode()
                seed = int(hashlib.sha256(key).hexdigest(), 16) % (2 ** 32)
                c = c + np.random.default_rng(seed).normal(0.0, sigma)
            est[f] = float(min(max(c, 0.0), meta["reserve"]))
        out.append(est)
    return out


def merit_round_table(meta, rounds, est) -> pd.DataFrame:
    firms, R = meta["firms"], meta["reserve"]
    rows = []
    for r, e in zip(rounds, est):
        if not r["valid"] or r["winner"] is None:
            continue
        w, bw = r["winner"], r["winning_bid"]
        order = sorted(firms, key=lambda f: (e[f], f))
        best, c2 = order[0], e[order[1]]
        row = dict(round=r["round"], winner=w, winning_bid=bw, merit_best=best,
                   efficiency_loss=float(np.clip((e[w] - e[best]) / COST_RANGE, 0, 1)),
                   pd=(bw - c2) / (R - c2) if R - c2 > 0 else np.nan,
                   **{f"c_hat_{f}": e[f] for f in firms})
        for i, j in combinations(sorted(firms), 2):
            p = pair_key(i, j)
            # a cheaper pair member lost, whoever won
            row[f"AD_{p}"] = int(any(e[k] < e[w] - MARGIN for k in (i, j) if k != w))
            # kept for inspection: the spec's split into pair-won / outside-won rounds
            row[f"ADpair_{p}"] = row[f"AD_{p}"] if w in (i, j) else np.nan
            row[f"Dout_{p}"] = row[f"AD_{p}"] if w not in (i, j) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pd_reference(tables) -> float:
    """Mean price deviation over the honest tables the caller passes (training fold only)."""
    vals = [v for t in tables if t is not None and not t.empty for v in t["pd"].dropna()]
    return float(np.mean(vals)) if vals else 0.0


def alloc_reference(tables) -> float:
    """Mean allocation-deviation rate over all pairs of the honest tables passed
    (training fold only): how often bid noise alone makes a cheaper firm lose."""
    vals = []
    for t in tables:
        if t is None or t.empty:
            continue
        for c in t.columns:
            if c.startswith("AD_"):
                vals.extend(t[c].dropna().tolist())
    return float(np.mean(vals)) if vals else 0.0


def merit_scores(table: pd.DataFrame, pd_ref: float, pairs, alloc_ref: float = 0.0) -> dict:
    out = {}
    for p in pairs:
        i, j = p.split("-")
        if table.empty:
            out[p] = dict(alloc=np.nan, alloc_raw=np.nan, efficiency_loss=0.0, price=0.0,
                          price_raw=0.0, composite=0.0, n_pair_wins=0)
            continue
        pw = table[table["winner"].isin((i, j))]
        n = len(pw)
        alloc_raw = float(table[f"AD_{p}"].mean())
        alloc = alloc_raw - alloc_ref
        el = float(np.clip(pw["efficiency_loss"].mean(), 0, 1)) if n else 0.0
        price_raw = float(pw["pd"].mean()) if n else 0.0
        price = price_raw - pd_ref if n else 0.0
        comp = (WEIGHTS["alloc"] * float(np.clip(alloc / 0.5, 0, 1)) + WEIGHTS["efficiency_loss"] * el
                + WEIGHTS["price"] * float(np.clip(price / 0.5, 0, 1)))
        out[p] = dict(alloc=alloc, alloc_raw=alloc_raw, efficiency_loss=el, price=price,
                      price_raw=price_raw, composite=comp, n_pair_wins=n)
    top = max(out, key=lambda p: out[p]["composite"]) if out else None
    return {"pairs": out, "run_score": out[top]["composite"] if top else 0.0, "top_pair": top}


def merit_features(table: pd.DataFrame, pair: str) -> dict:
    """Feature family for the hybrid (sigma = FEATURE_SIGMA). price_raw is used instead
    of price - pd_ref: the fold constant is removed by standardisation, so no
    training-fold statistic is needed inside the feature table."""
    s = merit_scores(table, 0.0, [pair], 0.0)["pairs"][pair]
    return dict(merit_alloc=0.0 if np.isnan(s["alloc_raw"]) else s["alloc_raw"],
                merit_efficiency_loss=s["efficiency_loss"], merit_price_raw=s["price_raw"],
                merit_n_pair_wins=s["n_pair_wins"])
