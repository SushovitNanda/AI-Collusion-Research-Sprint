"""Non-LLM feature set per pair of firms, in four families.

say          : message count, agreement claims, bid numbers mentioned, mean
               message length, trend of message length over rounds
do           : the three bid screens, pair win share, correlation of the two
               firms' bids, share of round-number bids
consistency  : cross-agent consistency, evidence weight, consistency score,
               self-executed rate, declaration contradictions (from claims)
internal     : public-statement length (mean, sd), similarity of the two
               firms' statements (Jaccard), bid precision (decimal places)

Nothing here encodes the condition, the label or the messaging mode.
"""
from __future__ import annotations

import re
from itertools import combinations

import numpy as np

from .claims import pair_key
from .screens import compute_screens

FAMILIES = {
    "say": ["msg_count", "agreement_claims", "bid_numbers_mentioned", "mean_msg_words",
            "msg_word_count_trend"],
    "do": ["S1", "S2", "S3", "pair_win_share", "bid_corr", "round_number_bids"],
    "consistency": ["cross_consistency", "evidence_weight", "consistency_score",
                    "self_executed_rate", "declaration_contradiction"],
    "internal": ["statement_words_mean", "statement_words_sd", "statement_similarity",
                 "bid_precision"],
}
ALL_FEATURES = [f for fam in FAMILIES.values() for f in fam]
_NUM = re.compile(r"\b\d{2,3}(?:\.\d+)?\b")


def _decimals(b) -> int:
    s = f"{b:.2f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def pair_features(meta, rounds, scored=None, claims=None) -> dict[str, dict]:
    firms = meta["firms"]
    valid = [r for r in rounds if r["valid"]]
    screens = compute_screens(valid, firms, meta["reserve"])
    out = {}
    for i, j in combinations(sorted(firms), 2):
        p = pair_key(i, j)
        msgs = [m for r in valid for m in r["chat"] if {m["from"], m["to"]} == {i, j}]
        words = [len(m["text"].split()) for m in msgs]
        per_round = {}
        for m in msgs:
            per_round[m["round"]] = per_round.get(m["round"], 0) + len(m["text"].split())
        if len(per_round) >= 2:
            xs, ys = zip(*sorted(per_round.items()))
            trend = float(np.polyfit(xs, ys, 1)[0])
        else:
            trend = 0.0
        pc = [c for c in (claims or []) if c["source"] == "message"
              and {c["sender"], c.get("recipient")} == {i, j}]
        if claims is not None:
            agreements = sum(c.get("stance") in ("propose", "accept") for c in pc)
            numbers = sum((c.get("sender_bid_value", -1) or -1) >= 0
                          or (c.get("recipient_bid_value", -1) or -1) >= 0 for c in pc)
        else:
            agreements = 0
            numbers = sum(len(_NUM.findall(m["text"])) for m in msgs)
        f = dict(msg_count=len(msgs), agreement_claims=agreements, bid_numbers_mentioned=numbers,
                 mean_msg_words=float(np.mean(words)) if words else 0.0,
                 msg_word_count_trend=trend)
        s = screens[p]
        bi = [r["bids"].get(i) for r in valid]
        bj = [r["bids"].get(j) for r in valid]
        both = [(a, b) for a, b in zip(bi, bj) if a is not None and b is not None]
        corr = 0.0
        if len(both) >= 3:
            a, b = np.array(both).T
            if a.std() > 0 and b.std() > 0:
                corr = float(np.corrcoef(a, b)[0, 1])
        allb = [b for b in bi + bj if b is not None]
        f.update(S1=s["S1"], S2=s["S2"], S3=s["S3"],
                 pair_win_share=float(np.mean([r["winner"] in (i, j) for r in valid])) if valid else 0.0,
                 bid_corr=corr,
                 round_number_bids=float(np.mean([abs(b / 5 - round(b / 5)) < 1e-9 for b in allb])) if allb else 0.0)
        c = (scored or {}).get("pairs", {}).get(p, {})
        f.update(cross_consistency=c.get("cross_consistency", 0.5), evidence_weight=c.get("evidence_weight", 0.0),
                 consistency_score=c.get("consistency_score", 0.0),
                 self_executed_rate=c.get("self_executed_rate", 0.0),
                 declaration_contradiction=c.get("declaration_contradiction", 0))
        st_i = [r["acts"][i].get("public_statement", "") or "" for r in valid]
        st_j = [r["acts"][j].get("public_statement", "") or "" for r in valid]
        lens = [len(s.split()) for s in st_i + st_j]
        f.update(statement_words_mean=float(np.mean(lens)) if lens else 0.0,
                 statement_words_sd=float(np.std(lens)) if lens else 0.0,
                 statement_similarity=float(np.mean([_jaccard(a, b) for a, b in zip(st_i, st_j)])) if valid else 0.0,
                 bid_precision=float(np.mean([_decimals(b) for b in allb])) if allb else 0.0,
                 cover_gap=s["cover_gap"] if s["cover_gap"] is not None else 0.0)
        out[p] = f
    return out


def run_features(pf: dict[str, dict], merit_table=None) -> dict:
    """Run-level features: the pair with the highest consistency score (ties ->
    larger cover gap). merit_table (sigma = 5) adds the privileged family, used
    only by the hybrid_plus_merit / hybrid_merit_only variants."""
    best = max(pf.items(), key=lambda kv: (kv[1]["consistency_score"], kv[1]["cover_gap"]))
    out = {"pair": best[0], **{k: best[1][k] for k in ALL_FEATURES}}
    if merit_table is not None:
        from .merit import merit_features
        out.update(merit_features(merit_table, best[0]))
    return out
