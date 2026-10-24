# results/model_panel — six more agent families (24 October 2026)

The rule ladder of the unscripted study at λ = 1 repeated with six further
model families reached through OpenRouter, plus explicit, tacit and honest
controls, scored by the same frozen GLM judge and the thresholds and hybrid
frozen on the pooled evaluation set (`results/extended/analysis/eval_pooled`).
Nothing in this folder entered any fit. Findings: [`RESULTS.md`](RESULTS.md);
every table: [`analysis/RESULTS_tables.md`](analysis/RESULTS_tables.md). File
formats: [`../README.md`](../README.md).

Families (config keys): `or_gptoss` GPT-OSS-120B, `or_gemini` Gemini 3.1 Flash
Lite, `or_deepseek` DeepSeek V4 Flash, `or_llama` Llama 4 Maverick,
`or_mistral` Mistral Small 2603, `or_haiku` Claude Haiku 5.5. Judge and
extractor ran on the second Modal workspace.

## runs/

| Folder | Runs | Contents |
|---|---|---|
| `panel_s1/` | 42 | per family, seed 1: `explicit_collusion`, `tacit_collusion`, `honest` (controls) and the four passages `dial` (law + declaration), `dial_norule_watched` (surveillance, no rule), `dial_softrule`, `dial_norule`, all at λ = 1 (honest at λ = 0). Judged with all five views |
| `panel_s23/` | 48 | the four passages at seeds 2–3 per family. Judged with the Say+Do view only (claims, consistency, screens and the hybrid cover every run) |
| `panel_all/` | 90 | **not versioned**: `panel_s1` + `panel_s23`, rebuilt by `bash scripts/pool_runs.sh`; the `--dial-logs` input of `analysis/panel` |

Mistral's provider rate-limited parallel runs; its 15 runs were deleted and
re-run one at a time (`step_logs/repair_deleted.txt`, `r_panel*_or_mistral.txt`).
Every run has at least 11 valid rounds. Three Say+Do judgements (DeepSeek,
seed 2) failed to parse and are missing.

## analysis/

| File / folder | Contents |
|---|---|
| `panel/` | `evaluate` output for the pooled evaluation set + `runs/panel_all` as `--dial-logs`: the panel runs appear in `runs.csv` (with `design_label`, no `label`) and `monitor_scores.csv`, scored with the frozen `thresholds.json` and hybrid |
| `panel_runs.csv` | one row per panel run: family, condition, seed, behavioural metrics, `cartel` (manipulation rule), every monitor's score, ring score, top pair and frozen-threshold flag |
| `panel_by_model_condition.csv` | means over seeds per family × condition |
| `panel_ladder.png` | Figure 2 of the report: collusion index and messages per run by family and passage |
| `panel_detection.png` | Figure 4 of the report: frozen-monitor flag rates per family on cartel and non-cartel runs |
| `panel_quotes.md` | first ring message or round-1 notes of every ladder and control run |
| `RESULTS_tables.md` | controls gate, ladder table, flag rates per family × condition, cross-family summary, monitor transfer table |

Produced by `python scripts/openrouter_panel.py runs/panel_all analysis/panel analysis`.

## step_logs/

`smoke_*` two-round checks per model (including the dropped free NVIDIA
model); `a_panel_*` / `a_panel_more_*` agent grids (seed 1 / seeds 2–3);
`b_monitor_s1` the five-view judging of seed 1 (`b_monitor_s23` was stopped
after a few calls to switch to the Say+Do view); `r_*` the finish pass
(`scripts/panel_repair.sh`): Mistral re-runs, monitors, pooled evaluate,
panel summary; `launcher.txt`, `finish.txt`, `mistral_guard.txt`,
`repair_deleted.txt` operational logs; `../progress.txt` phase markers.
