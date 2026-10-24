# results/extended — second batch, extensions, replication (24 October 2026)

Adds to `results/main` without changing it: a second evaluation batch with new
cost draws and the pooled 48-run analysis, three extensions of the unscripted
study, and a second-agent-family replication. Agents ran on a second Modal
workspace with identical deployments; judging (GLM-5.3) and claim extraction
(Qwen) stayed on the first. Pooling rule, fixed before batch 2 was run: batch 2
is reported per batch and pooled with batch 1, never chosen over it.
Findings: [`RESULTS.md`](RESULTS.md). File formats: [`../README.md`](../README.md).

## runs/

| Folder | Runs | Contents |
|---|---|---|
| `eval_b2/` | 24 | evaluation set, batch 2: `explicit_collusion` and `tacit_collusion` seeds 7–12; the four negatives seeds 7–9 |
| `ext/` | 8 | extensions at λ = 1: `dial_softrule` ("the Council expects independent bids", no law) seeds 1–3; `dial_norule_watched` (no rule, "the regulator reads all messages") seeds 1–3; two no-rule up/down sweeps with a 12-round tail at λ = 0 (`updown...-tail12`, seeds 1–2) |
| `dial_norule_all/` | 10 | the no-rule arm pooled for the persistence analysis: the 6 fixed runs and 2 sweeps copied from `results/main/runs/dial_norule` plus the 2 tail sweeps from `ext/` |
| `robustness_glm/` | 2 | GLM-5.3 as the agents (reasoning forced on), rule vs no rule at λ = 1, seed 1, judged by Qwen |
| `eval_pooled/` | 48 | **not versioned**: `results/main/runs/eval` + `eval_b2`, rebuilt by `bash scripts/pool_runs.sh`; the `--logs` input of every pooled analysis |

## analysis/

| Folder | `evaluate` inputs | Use in the report |
|---|---|---|
| `eval_b2/` | `runs/eval_b2` | batch 2 alone (Table 6 of the report) |
| `eval_pooled/` | `runs/eval_pooled` | **the headline monitor comparison** (48 runs): Tables 2–3, the Δ test, `thresholds.json` used for every dial and panel run |
| `main_rule/` | pooled + `results/main/runs/dial` | rule arm with pooled thresholds |
| `main_norule/` | pooled + `results/main/runs/dial_norule` | no-rule arm with pooled thresholds |
| `main_norule_all/` | pooled + `runs/dial_norule_all` | four-sweep hysteresis and tail persistence (Figure 3 of the report) |
| `main_ext/` | pooled + `runs/ext` | soft rule and surveillance notice (Table 5 of the report) |
| `robustness_glm/` | — | `README.md` only: `evaluate` requires one judge model and this subset is judged by Qwen; its numbers are computed from the run folders (see `step_logs/c2_go_no_go_robustness.txt`) |

## step_logs/

`a1`–`a3` grids (batch 2, extensions, GLM replication); `b0`–`b2` monitors;
`c1` evaluate (all analyses); `c2` go/no-go per set; `c3` transcripts;
`launcher.txt` and `progress.txt` are the pipeline's own log and phase markers.
