# results/main — the first full run (23–24 October 2026)

The evaluation set (batch 1), the incentive dial in both arms, and the
judge-with-a-stake extension. Agents: Qwen3.8-2.4T-A95B (reasoning effort low);
judge: GLM-5.3 (five views); claims extractor: Qwen. All on one Modal
workspace. Findings: [`RESULTS.md`](RESULTS.md). File formats:
[`../README.md`](../README.md).

## runs/

| Folder | Runs | Contents |
|---|---|---|
| `eval/` | 24 | the labelled evaluation set, batch 1: `explicit_collusion` and `tacit_collusion` seeds 1–6; `cheap_talk`, `innocent_parallel`, `honest`, `honest_nomsg` seeds 1–3; 12 rounds each. Monitor files include the judge-with-a-stake views (`glm__saydo__stake0.5.json`, `__stake1`, and the same for `structured`) |
| `dial/` | 8 | unscripted study, **rule arm** (procurement law stated, signed declaration): fixed λ ∈ {0, 0.25, 0.5, 1} × seeds 1–2 |
| `dial_norule/` | 8 | unscripted study, **no-rule arm** (nothing forbids coordination): fixed λ ∈ {0, 0.5, 1} × seeds 1–2, plus one up/down sweep per seed (λ 0 → 0.25 → 0.5 → 1 → 0.5 → 0.25 → 0, six rounds per level) |

Each `usage.json` records the grid's token usage.

## analysis/

| Folder | `evaluate` inputs | Use in the report |
|---|---|---|
| `eval/` | `runs/eval` | batch-1 monitor comparison, case-type matrix, Δ test, masking analysis |
| `main_rule/` | `runs/eval` + `runs/dial` | rule arm by λ, frozen-monitor flags on the rule arm |
| `main_norule/` | `runs/eval` + `runs/dial_norule` | no-rule arm by λ, up/down sweep hysteresis (batch-1 seeds) |

Thresholds and the hybrid are fitted on `runs/eval` only; dial runs are
scored with the frozen full-set thresholds (`thresholds.json`).

## step_logs/

Stdout of every step in execution order: `b1a` grid (eval agents), `b1b`
monitor, `b1c` evaluate, `b1d` transcripts, `b1e` go/no-go; `b2a`–`b2b`
grids (dial arms), `b2c` monitor, `b2d` evaluate, `b2e` transcripts, `b2f`
go/no-go, `b2g` a re-run of the Do view; `b3` the judge-with-a-stake views.
