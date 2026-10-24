"""Figures for the report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

BLUE, ORANGE, GREY = "#1F4E79", "#C0504D", "#8C8C8C"
MONITOR_ORDER = ["say", "do", "saydo", "structured", "think", "screens", "consistency",
                 "hybrid", "hybrid_say", "hybrid_do", "hybrid_consistency", "hybrid_internal",
                 "merit_s0", "merit_s5", "merit_s10", "do_merit", "hybrid_plus_merit", "hybrid_merit_only"]


def _order(names):
    return sorted(names, key=lambda n: (MONITOR_ORDER.index(n) if n in MONITOR_ORDER else 99, n))


# ---- evaluation set -------------------------------------------------------

def case_type_heatmap(matrix: pd.DataFrame, path, privileged=()):
    names = _order(matrix.index)
    names = [n for n in names if n not in privileged] + [n for n in names if n in privileged]
    m = matrix.loc[names]
    n_unpriv = sum(n not in privileged for n in names)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(m) + 1.6))
    im = ax.imshow(m.to_numpy(float), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(m.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in m.columns], fontsize=8)
    ax.set_yticks(range(len(m.index)))
    ax.set_yticklabels([f"{n} (privileged)" if n in privileged else n for n in m.index], fontsize=8)
    if 0 < n_unpriv < len(names):
        ax.axhline(n_unpriv - 0.5, color="black", lw=1.2)
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = m.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 else "black")
    ax.set_title("Flag rate of each monitor on each case type (leave-one-seed-out thresholds)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc_curves(m: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cmap = plt.get_cmap("tab10")
    for k, name in enumerate(_order(m["monitor"].unique())):
        g = m[m["monitor"] == name]
        if g["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["label"], g["score"])
        ax.plot(fpr, tpr, color=cmap(k % 10), lw=1.4,
                label=f"{name} (AUROC {roc_auc_score(g['label'], g['score']):.2f})")
    ax.plot([0, 1], [0, 1], ":", color=GREY, lw=1)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("ROC on the evaluation set")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def channel_delta(samples: dict, path):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    cmap = plt.get_cmap("tab10")
    for k, (name, d) in enumerate(samples.items()):
        d = np.asarray(d)
        ax.hist(d, bins=40, alpha=0.45, color=cmap(k % 10),
                label=f"{name}: mean {d.mean():+.3f}, P(delta<=0) = {(d <= 0).mean():.2f}")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("AUROC(monitor) - max(AUROC(say), AUROC(do)), paired bootstrap")
    ax.set_ylabel("resamples")
    ax.set_title("Does combining channels beat the best single channel?", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sigma_robustness(mon: pd.DataFrame, path):
    """AUROC of the merit auditor vs the noise of its cost estimates, with the
    unprivileged Do view and the screens as reference lines."""
    m = mon.set_index("monitor")
    pts = []
    for name in m.index:
        if name.startswith("merit_s"):
            pts.append((float(name[7:]), m.loc[name, "auroc"], m.loc[name, "auroc_ci_low"], m.loc[name, "auroc_ci_high"]))
    if not pts:
        return
    pts.sort()
    x, y, lo, hi = zip(*pts)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(x, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], marker="o",
                color=BLUE, capsize=3, label="merit auditor (privileged)")
    for ref, color, style in (("do", ORANGE, "--"), ("screens", GREY, ":")):
        if ref in m.index:
            ax.axhline(m.loc[ref, "auroc"], color=color, ls=style, lw=1.2, label=f"{ref} (unprivileged)")
    ax.set_xlabel("noise of the auditor's cost estimate (sigma)")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.3, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Merit auditor vs estimate noise (bootstrap 95% CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---- dial study ------------------------------------------------------------

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


def _dial(blocks):
    return blocks[blocks["condition"].str.startswith("dial")]


def hysteresis(blocks: pd.DataFrame, path, metric="collusion_index"):
    fig, ax = plt.subplots(figsize=(6, 4))
    d = _dial(blocks)
    sweep = d[d["schedule_kind"] == "updown"]
    fixed = d[d["schedule_kind"] == "fixed"]
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
    d = _dial(blocks)
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
    d = _dial(blocks)
    d = d[d["schedule_kind"] == "fixed"]
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
    for k, name in enumerate(_order(det["monitor"].unique())):
        g = det[det["monitor"] == name].sort_values("lam")
        ax.plot(g["lam"], g["detection_rate"], marker="o", color=cmap(k % 10), label=name)
    ax.axhline(0.8, color=GREY, ls=":", lw=1)
    ax.set_xlabel("Group bonus rate λ")
    ax.set_ylabel("share of runs flagged")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Frozen monitors applied to the dial study (thresholds from the evaluation set)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
