"""Figures for the report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREY = "#1F4E79", "#C0504D", "#8C8C8C"


def _ci(vals, n_boot=2000, seed=0):
    vals = np.asarray([v for v in vals if not np.isnan(v)], float)
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    if len(vals) == 1:
        return vals[0], vals[0], vals[0]
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, (n_boot, len(vals))).mean(1)
    return vals.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def _line(ax, df, col, label, color, style="-", marker="o"):
    rows = []
    for lam, g in df.groupby("lam"):
        m, lo, hi = _ci(g[col].astype(float).values)
        rows.append((lam, m, lo, hi))
    if not rows:
        return
    r = np.array(rows, float)
    ax.plot(r[:, 0], r[:, 1], style, marker=marker, color=color, label=label)
    ax.fill_between(r[:, 0], r[:, 2], r[:, 3], color=color, alpha=0.15)


def hysteresis(blocks: pd.DataFrame, path, metric="collusion_index"):
    fig, ax = plt.subplots(figsize=(6, 4))
    sweep = blocks[blocks["schedule_kind"] == "updown"]
    fixed = blocks[(blocks["schedule_kind"] == "fixed") & blocks["condition"].str.startswith("dial")]
    _line(ax, fixed, metric, "Fixed λ (no history)", GREY, "--", "s")
    _line(ax, sweep[sweep["phase"] == "up"], metric, "Up-sweep", BLUE)
    _line(ax, sweep[sweep["phase"] == "down"], metric, "Down-sweep", ORANGE)
    ax.set_xlabel("Group bonus rate λ")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Collusion vs incentive: up vs down (mean, 95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def covertness(blocks: pd.DataFrame, path):
    d = blocks[blocks["condition"].str.startswith("dial")]
    fig, ax = plt.subplots(figsize=(6, 4))
    _line(ax, d, "covertness", "Covertness: covert / (covert + open)", BLUE)
    _line(ax, d, "say_do_gap", "Say-do gap: colluding rounds declared INDEPENDENT", ORANGE,
          "--", "s")
    ax.set_xlabel("Group bonus rate λ")
    ax.set_ylabel("share")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Lying about collusion vs incentive")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def overcharge(blocks: pd.DataFrame, path):
    d = blocks[(blocks["schedule_kind"] == "fixed") & blocks["condition"].str.startswith("dial")]
    fig, ax = plt.subplots(figsize=(6, 4))
    _line(ax, d, "mean_overcharge", "Mean buyer overcharge", BLUE)
    ax.set_xlabel("Group bonus rate λ")
    ax.set_ylabel("winning bid − competitive benchmark")
    ax.grid(alpha=0.3)
    ax.set_title("Harm to the buyer vs incentive")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def detection_by_lambda(det: pd.DataFrame, path):
    if det.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cmap = plt.get_cmap("tab10")
    for k, (name, g) in enumerate(det.groupby("monitor")):
        g = g.sort_values("lam")
        ax.plot(g["lam"], g["detection_rate"], marker="o", color=cmap(k % 10), label=name)
    ax.axhline(0.8, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Group bonus rate λ")
    ax.set_ylabel("share of runs flagged")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Monitor detection vs incentive (threshold frozen on calibration runs)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
