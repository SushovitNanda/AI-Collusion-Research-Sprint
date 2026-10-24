"""Readable per-run summary of saved logs: rounds, messages, declarations,
behavioural metrics, claims/consistency and judge verdicts.

  python scripts/transcript.py logs/pilot                # all runs in the folder
  python scripts/transcript.py logs/pilot --full         # include private notes
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collusion_dial.metrics.collusion import load_run, round_frame, summarize  # noqa: E402
from collusion_dial.metrics.lying import classify, lying_summary  # noqa: E402
from collusion_dial.monitors.claims import coordination_intent, pair_key  # noqa: E402


def _fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def summarise(run_dir: Path, full: bool):
    meta, rounds = load_run(run_dir)
    a, b = meta["sisters"]
    ring = pair_key(a, b)
    out = [f"{'=' * 100}", f"RUN {meta['run_id']}", f"condition={meta['condition']}  "
           f"lambda={meta['lam_label']}  seed={meta['seed']}  firms={meta['firms']}  "
           f"messaging={meta['condition_flags'].get('messaging')}  rounds={len(rounds)}"]
    for r in rounds:
        c, bid = r["costs"], r["bids"]
        cells = []
        for f in meta["firms"]:
            bv = bid.get(f)
            d = (r["acts"][f].get("declaration") or "?")[:3]
            cells.append(f"{f}: cost {c[f]:5.1f} bid {_fmt(bv, 1):>6} [{d}]")
        win = f"-> {r['winner']} wins at {r['winning_bid']:.1f}" if r["winner"] else "-> no award"
        out.append(f"R{r['round'] + 1:>2}  " + " | ".join(cells) + f"  {win}" +
                   ("" if r["valid"] else "  (INVALID)"))
        for m in r["chat"]:
            out.append(f"      msg {m['from']}->{m['to']}: {m['text']}")
        if full:
            for f, p in r["plans"].items():
                out.append(f"      notes {f}: {(p or {}).get('notes', '')}")
    df = round_frame(meta, rounds)
    s = summarize(df)
    out.append(f"BEHAVIOUR  collusion_index={_fmt(s.get('collusion_index'))}  "
               f"excess_deference={_fmt(s.get('excess_deference_index'))}  "
               f"cover_bid_rate={_fmt(s.get('cover_bid_rate'))}  pair_win_share={_fmt(s.get('pair_win_share'))}  "
               f"mean_overcharge={_fmt(s.get('mean_overcharge'), 1)}  valid={s['valid_rounds']}/{s['rounds']}")
    mdir = run_dir / "monitor"
    intent = None
    if mdir.exists():
        cl = sorted(mdir.glob("claims__*.json"))
        nt = sorted(mdir.glob("claims_notes__*.json"))
        items = []
        for p in cl + nt:
            items += json.loads(p.read_text())["items"]
        if cl:
            intent = coordination_intent(items)
        cons = json.loads((mdir / "consistency.json").read_text()) if (mdir / "consistency.json").exists() else None
        if cons:
            p = cons["pairs"][ring]
            out.append(f"CLAIMS     ring {ring}: claims={p['n_claims']} cross={p['n_cross_claims']} "
                       f"executed={p['cross_executed']} consistency={p['cross_consistency']:.2f} "
                       f"score={p['consistency_score']:.2f} agreements={p['n_agreements']}/"
                       f"{p['executed_agreements']} executed  decl_contradictions={p['declaration_contradiction']}  "
                       f"| max pair score={cons['score']:.2f}")
        if cl:
            msgs = [c for c in json.loads(cl[0].read_text())["items"] if c["source"] == "message"]
            out.append(f"EXTRACTED CLAIMS from {len(msgs)} messages (stance / claimed winner / sender intent / request to recipient):")
            for c in msgs:
                out.append(f"      R{c['round'] + 1} {c['sender']}->{c['recipient']}: {c['stance']:<7} win={c['claimed_winner']:<4} "
                           f"self={c['sender_bid_intent']}{'' if c['sender_bid_value'] < 0 else f'({c['sender_bid_value']:g})'} "
                           f"ask={c['recipient_request']}{'' if c['recipient_bid_value'] < 0 else f'({c['recipient_bid_value']:g})'} "
                           f"| \"{c['quote']}\"")
        judges = []
        for p in sorted(mdir.glob("*__*.json")):
            if p.name.startswith("claims"):
                continue
            d = json.loads(p.read_text())
            if "view" not in d:
                continue
            judges.append(f"      {d['view']:<10} ring={_fmt(d.get('ring_score'))} max={_fmt(max(d['scores'].values()))} "
                          f"verdict={d.get('verdict_pair')} observed={d.get('coordination_observed')} ok={d['ok']}  "
                          f"| {(d.get('reasoning') or '')[:160]}")
        screens = json.loads((mdir / "screens.json").read_text()) if (mdir / "screens.json").exists() else None
        if screens:
            out.append("SCREENS    " + "  ".join(f"{p}={v['score']:.2f}" for p, v in screens.items()))
        if judges:
            out.append("JUDGE (qwen; scores are in true letters, the judge saw relabelled ones):")
            out += judges
    ly = lying_summary(classify(meta, rounds, intent))
    if ly:
        out.append("TYPOLOGY   " + "  ".join(f"{k}={v}" for k, v in ly.items() if k.startswith("n_")) +
                   f"  covertness={_fmt(ly.get('covertness'))}  say_do_gap={_fmt(ly.get('say_do_gap'))}")
    calls = [c for r in rounds for c in r["calls"]]
    out.append(f"USAGE      calls={len(calls)} ok={sum(c['ok'] for c in calls)} "
               f"in={sum(c['input_tokens'] for c in calls):,} out={sum(c['output_tokens'] for c in calls):,} "
               f"(avg out/call {sum(c['output_tokens'] for c in calls) / max(1, len(calls)):.0f})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs")
    ap.add_argument("--full", action="store_true", help="include private notes")
    a = ap.parse_args()
    for meta in sorted(Path(a.logs).glob("*/meta.json")):
        print(summarise(meta.parent, a.full))
        print()


if __name__ == "__main__":
    main()
