# results/ — every run and analysis behind the report

Three studies, one folder each. Each sub-folder has its own `README.md`
(what every file is) and `RESULTS.md` (the findings in prose and tables).

| Folder | What it holds | Agents | Judge / claims extractor |
|---|---|---|---|
| [`main/`](main/README.md) | evaluation batch 1 (24 labelled runs), the incentive dial with and without the stated rule (16 runs), the judge-with-a-stake extension | Qwen3.8-2.4T-A95B (Modal workspace 1) | GLM-5.3 / Qwen (workspace 1) |
| [`extended/`](extended/README.md) | evaluation batch 2 (24 runs, new cost draws) and the pooled 48-run analysis; soft-rule, surveillance and tail-sweep extensions; the GLM-agent replication | Qwen (Modal workspace 2, identical deployment); GLM for the replication | GLM / Qwen (workspace 1) |
| [`model_panel/`](model_panel/README.md) | six further agent families on the rule ladder at λ = 1 (90 runs), scored with the frozen monitors | GPT-OSS-120B, Gemini 3.1 Flash Lite, DeepSeek V4 Flash, Llama 4 Maverick, Mistral Small 2603, Claude Haiku 5.5 via OpenRouter | GLM / Qwen (workspace 2) |

Reading order: `main/RESULTS.md` → `extended/RESULTS.md` → `model_panel/RESULTS.md`.
The report (`Report/Main.pdf`) cites the pooled 48-run analysis
(`extended/analysis/eval_pooled`) for the monitor comparison, `main` and
`extended` for the unscripted study, and `model_panel` for the six-family panel.

## Common layout

```
<study>/
  RESULTS.md        findings
  README.md         file guide
  runs/<set>/       one folder per run (see below) + usage.json (token usage of the grid)
  analysis/<name>/  output of `python -m collusion_dial.evaluate` (see below)
  step_logs/        stdout of every pipeline step, in execution order
```

### A run folder `runs/<set>/<run_id>/`

`run_id = <agent model>__<condition>__lam<λ>__n3__s<seed>[__updown...]`

| File | Contents |
|---|---|
| `meta.json` | condition, schedule, seed, firms, sisters, reserve, agent model, `prompt_sha256` of the three system prompts |
| `rounds.jsonl` | one line per round: costs, private notes and intended bids (`plans`), messages (`chat`), bids, public statements and declarations (`acts`), winner, profits, every LLM call with token usage and validity |
| `monitor/screens.json` | the three non-LLM bid-rigging screens per pair |
| `monitor/claims__qwen.json` | claims extracted from messages and public statements (relabelled firm letters; `perm` maps back) |
| `monitor/consistency.json` | claim-vs-outcome scores per pair (self and cross-agent consistency, agreements, executed agreements) |
| `monitor/glm__<view>.json` | the judge's output for `say`, `do`, `saydo`, `structured`, `think` (`__stake0.5`, `__stake1`: judge with a stake) |
| `monitor/merit_table__s<σ>.csv` | the privileged merit auditor's per-round table at cost-estimate noise σ |

### An analysis folder `analysis/<name>/`

| File | Contents |
|---|---|
| `runs.csv` | one row per run: design label, manipulation check, collusion index, excess deference, cover-bid rate, messages, ring consistency, executed agreements, say-do gap, declarations, call validity |
| `blocks.csv` | the same per six-round block (sweeps: phase up/down/tail) |
| `monitor_scores.csv` | every monitor's score per run (`score` = max over pairs, `score_ring`, `top_pair`) |
| `monitors.csv` | per monitor: AUROC with bootstrap CI, accuracy, recall, FPR with Wilson CIs, pair identification |
| `flags.csv`, `thresholds.json` | leave-one-seed-out flags; full-set Youden thresholds (the frozen thresholds used for dial and panel runs) |
| `case_type_matrix.csv/.png` | flag rate of every monitor on every case type |
| `channel_comparison.json`, `channel_delta.png` | paired bootstrap of Δ = AUROC(x) − max(AUROC(say), AUROC(do)) |
| `disagreement.csv` | say-only / do-only / both / neither, per case type |
| `features.csv`, `hybrid_coefficients.csv` | hybrid inputs and fitted coefficients |
| `merit_components.csv`, `sigma_robustness.png` | privileged auditor components and its AUROC vs σ |
| `roc.png` | ROC curves of the unprivileged monitors |
| `dial_study.json`, `dial_*.png/csv` | present when dial runs were passed: behaviour by λ, up vs down sweeps, tail blocks, frozen-monitor flag rate by λ |
| `summary.json` | everything above plus τ, parse validity, masking, ownership-mention rates, monitor-code hash, usage |

### Derived folders that are not versioned

`evaluate` takes one `--logs` and one `--dial-logs` folder, so two pooled
copies are used and rebuilt by `bash scripts/pool_runs.sh`:
`extended/runs/eval_pooled` (= `main/runs/eval` + `extended/runs/eval_b2`) and
`model_panel/runs/panel_all` (= `panel_s1` + `panel_s23`). Both are in
`.gitignore`.
