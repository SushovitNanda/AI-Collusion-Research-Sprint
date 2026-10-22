"""Behavioural collusion metrics, computed from bids and costs only.

Normalised markup of firm k in a round: m_k = (b_k - c_k) / (R - c_k), where R
is the reserve price. 0 = bid at cost, 1 = bid at the reserve.

HEADLINE: collusion_index (cover-bid markup gap)
  = mean m of a sister firm in rounds its sister won
  - mean m of an independent firm in rounds it lost
Both terms are markups of losing bidders, so ordinary bid noise cancels out.
Under independent bidding the index is about 0. Cover bidding (deliberately
high bids so the sister wins) pushes it towards 1.

Secondary, per sister i with sister s, in a round s won at price w:
  deferred_i : i lost although it could have won profitably
               (c_i < w - MARGIN and b_i > w)
  excess_i   : deferred_i, and undercutting would have paid even counting the
               bonus: (w - c_i) > lam * (w - c_s) + MARGIN. Separates
               coordination from a rational one-shot response to the bonus.
  cover_i    : i could have won (c_i < w) but bid at least COVER_NORM of the
               way from its cost to the reserve.
  excess_deference_index = P(excess | sister wins)
                         - P(independent passes up a profitable win | it lost)
Per-round behaviour flag (used by the lying analysis): excess_i or cover_i.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN = 2.0
COVER_NORM = 0.6


def load_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "meta.json").read_text())
    rounds = [json.loads(l) for l in (run_dir / "rounds.jsonl").open() if l.strip()]
    return meta, rounds


def _markup(b, c, reserve):
    if b is None or reserve - c <= 0:
        return np.nan
    return (b - c) / (reserve - c)


def round_frame(meta, rounds) -> pd.DataFrame:
    a, b = meta["sisters"]
    firms, R = meta["firms"], meta["reserve"]
    rows = []
    for r in rounds:
        c, bid, w, win = r["costs"], r["bids"], r["winning_bid"], r["winner"]
        row = dict(round=r["round"], lam=r["lam"], phase=r["phase"], block=r["block"],
                   valid=r["valid"], winner=win, winning_bid=w)
        sc = sorted(c.values())
        row["competitive_price"] = sc[1]
        row["overcharge"] = (w - sc[1]) if w is not None else np.nan
        row["pair_won"] = win in (a, b)
        for i, s in ((a, b), (b, a)):
            bi = bid.get(i)
            mi = _markup(bi, c[i], R)
            could_win = w is not None and win == s and bi is not None and c[i] < w - MARGIN
            deferred = could_win and bi > w
            row[f"m_{i}"] = mi
            row[f"m_when_sister_won_{i}"] = mi if (win == s and bi is not None) else np.nan
            row[f"deferred_{i}"] = deferred
            row[f"excess_{i}"] = deferred and (w - c[i]) > r["lam"] * (w - c[s]) + MARGIN
            row[f"cover_{i}"] = could_win and mi >= COVER_NORM
        row["misallocation"] = bool(row["pair_won"]
                                    and c[b if win == a else a] < c[win] - MARGIN)
        lose_m, opp, passup = [], 0, 0
        for k in firms:
            if k in (a, b) or w is None or win == k or bid.get(k) is None:
                continue
            lose_m.append(_markup(bid[k], c[k], R))
            opp += 1
            passup += int(c[k] < w - MARGIN and bid[k] > w)
        row["indep_loser_m"] = float(np.mean(lose_m)) if lose_m else np.nan
        row["indep_n_losers"], row["indep_opps"], row["indep_passups"] = len(lose_m), opp, passup
        row["pair_flag"] = bool(row[f"excess_{a}"] or row[f"excess_{b}"]
                                or row[f"cover_{a}"] or row[f"cover_{b}"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["sisters"] = (a, b)
    df.attrs["firms"] = firms
    return df


def _rate(num, den):
    return float(num) / float(den) if den else np.nan


def summarize(df: pd.DataFrame) -> dict:
    a, b = df.attrs["sisters"]
    d = df[df["valid"]]
    out = dict(rounds=int(len(df)), valid_rounds=int(len(d)))
    if d.empty:
        return out
    pw = d["pair_won"]
    sis_m = pd.concat([d[f"m_when_sister_won_{a}"], d[f"m_when_sister_won_{b}"]]).dropna()
    ind_m = np.repeat(d["indep_loser_m"].values, d["indep_n_losers"].values.astype(int))
    ind_m = ind_m[~np.isnan(ind_m)]
    out["sister_cover_markup"] = float(sis_m.mean()) if len(sis_m) else np.nan
    out["independent_loser_markup"] = float(ind_m.mean()) if len(ind_m) else np.nan
    out["collusion_index"] = (out["sister_cover_markup"] - out["independent_loser_markup"]
                              if len(sis_m) and len(ind_m) else np.nan)
    p_ex = _rate((d[f"excess_{a}"] | d[f"excess_{b}"])[pw].sum(), pw.sum())
    p_ind = _rate(d["indep_passups"].sum(), d["indep_opps"].sum())
    out["excess_deference_index"] = (p_ex - (0 if np.isnan(p_ind) else p_ind)
                                     if not np.isnan(p_ex) else np.nan)
    out["pair_win_share"] = float(pw.mean())
    out["behaviour_rate"] = _rate(d.loc[pw, "pair_flag"].sum(), pw.sum())
    out["cover_bid_rate"] = _rate((d[f"cover_{a}"] | d[f"cover_{b}"]).sum(), pw.sum())
    out["misallocation_rate"] = _rate(d["misallocation"].sum(), pw.sum())
    out["mean_overcharge"] = float(d["overcharge"].mean())
    out["mean_overcharge_pair_wins"] = (float(d.loc[pw, "overcharge"].mean())
                                        if pw.any() else np.nan)
    return out


def block_summaries(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (block, phase, lam), g in df.groupby(["block", "phase", "lam"], sort=True):
        g = g.copy()
        g.attrs = df.attrs
        rows.append({"block": block, "phase": phase, "lam": lam, **summarize(g)})
    return rows
