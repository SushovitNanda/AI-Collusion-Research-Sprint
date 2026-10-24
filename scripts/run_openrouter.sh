#!/usr/bin/env bash
# OpenRouter model panel: agents from six model families on OpenRouter, scored by
# the frozen GLM judge (Modal workspace 3) with Qwen claim extraction.
#
#   phase A1  seed-1 panel per model (7 runs: explicit, tacit, honest, 4 rule-ladder arms)
#   phase A2  seeds 2-3 of the rule ladder (8 runs per model), in a second folder so
#             that judging of A1 can start while A2 agents are still running
#   phase B1/B2  monitors on each folder
#   phase C   pooled folder -> evaluate (frozen thresholds from the pooled eval set)
#             -> panel summary (CSV, plots, RESULTS tables)
#
# Re-running the script is safe: finished runs and existing monitor files are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
OUT=results/model_panel
S1=$OUT/runs/panel_s1
S23=$OUT/runs/panel_s23
ALL=$OUT/runs/panel_all
LOGS=$OUT/step_logs
CFG=config/models.yaml
PROG=$OUT/progress.txt
mkdir -p "$S1" "$S23" "$LOGS" "$OUT/analysis"
note() { echo "$(date '+%F %T') $*" | tee -a "$PROG"; }

MODELS="or_gptoss or_gemini or_deepseek or_llama or_mistral or_haiku"
# upstream pools rate-limit bursts: Mistral one run at a time, Llama two
par() { case "$1" in or_mistral) echo 1;; or_llama) echo 2;; *) echo 4;; esac; }

agents() {   # $1 plan, $2 out folder
  local plan=$1 out=$2 pids=()
  note "AGENTS START plan=$plan out=$out"
  for m in $MODELS; do
    python -m collusion_dial.grid --plan "$plan" --model "$m" --config "$CFG" --out "$out" \
      --parallel-runs "$(par "$m")" --concurrency 6 --yes > "$LOGS/a_${plan}_${m}.txt" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  note "AGENTS DONE plan=$plan out=$out"
}

monitor() {  # $1 folder, $2 tag, remaining args go to monitor (e.g. --views saydo)
  local folder=$1 tag=$2; shift 2
  note "MONITOR START $tag"
  python -m collusion_dial.monitor --logs "$folder" --judge-model glm --extractor-model qwen \
    --config "$CFG" --concurrency 6 "$@" > "$LOGS/b_monitor_$tag.txt" 2>&1
  note "MONITOR DONE $tag"
}

note "PIPELINE START"
agents panel "$S1"
# judge seed 1 while the seed 2-3 agents run
monitor "$S1" s1 &
MON1=$!
agents panel_more "$S23"
wait $MON1
monitor "$S23" s23 --views saydo     # seeds 2-3: claims + the Say+Do view only (budget)

note "ANALYSIS START"
bash scripts/pool_runs.sh > "$LOGS/c_pool.txt" 2>&1
python -m collusion_dial.evaluate --logs results/extended/runs/eval_pooled --dial-logs "$ALL" \
  --out "$OUT/analysis/panel" --judge glm --claims-model qwen > "$LOGS/c_evaluate.txt" 2>&1
python scripts/openrouter_panel.py "$ALL" "$OUT/analysis/panel" "$OUT/analysis" > "$LOGS/c_panel.txt" 2>&1
note "PIPELINE DONE"
