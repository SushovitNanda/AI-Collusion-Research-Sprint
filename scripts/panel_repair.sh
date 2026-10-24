#!/usr/bin/env bash
# Repair pass for the OpenRouter panel, run after scripts/run_openrouter.sh:
#   1. delete every panel run with more than one invalid round (the >= 11/12 gate)
#   2. re-run the missing runs of each model ONE AT A TIME (upstream rate limits,
#      e.g. Mistral's shared pool, broke runs at 4 parallel runs)
#   3. judge the new runs (existing monitor files are kept), re-pool, evaluate, summarise
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
OUT=results/model_panel; S1=$OUT/runs/panel_s1; S23=$OUT/runs/panel_s23; ALL=$OUT/runs/panel_all
LOGS=$OUT/step_logs; CFG=config/models.yaml; PROG=$OUT/progress.txt
note() { echo "$(date '+%F %T') $*" | tee -a "$PROG"; }
MODELS=${MODELS:-"or_gptoss or_gemini or_deepseek or_llama or_mistral or_haiku"}

note "REPAIR START"
python - <<'EOF' | tee -a "$LOGS/repair_deleted.txt"
import json, shutil
from pathlib import Path
for folder in ("results/model_panel/runs/panel_s1", "results/model_panel/runs/panel_s23"):
    for rj in sorted(Path(folder).glob("*/rounds.jsonl")):
        rows = [json.loads(l) for l in rj.open(encoding="utf8")]
        bad = sum(1 for r in rows if not r.get("valid", True))
        if bad > 1 or len(rows) < 12:
            shutil.rmtree(rj.parent)
            print(f"deleted {rj.parent} (rounds={len(rows)}, invalid={bad})")
EOF
for m in $MODELS; do
  python -m collusion_dial.grid --plan panel --model "$m" --config "$CFG" --out "$S1" \
    --parallel-runs 1 --concurrency 3 --yes > "$LOGS/r_panel_${m}.txt" 2>&1
  python -m collusion_dial.grid --plan panel_more --model "$m" --config "$CFG" --out "$S23" \
    --parallel-runs 1 --concurrency 3 --yes > "$LOGS/r_panel_more_${m}.txt" 2>&1
  note "REPAIR AGENTS DONE $m"
done
note "REPAIR MONITOR START"
python -m collusion_dial.monitor --logs "$S1" --judge-model glm --extractor-model qwen --config "$CFG" --concurrency 6 > "$LOGS/r_monitor_s1.txt" 2>&1
# seeds 2-3: claims (consistency, hybrid) + the combined Say+Do judge view only (Modal budget)
python -m collusion_dial.monitor --logs "$S23" --judge-model glm --extractor-model qwen --config "$CFG" --concurrency 6 --views saydo > "$LOGS/r_monitor_s23.txt" 2>&1
note "REPAIR MONITOR DONE"
bash scripts/pool_runs.sh > "$LOGS/r_pool.txt" 2>&1
python -m collusion_dial.evaluate --logs results/extended/runs/eval_pooled --dial-logs "$ALL" \
  --out "$OUT/analysis/panel" --judge glm --claims-model qwen > "$LOGS/r_evaluate.txt" 2>&1
python scripts/openrouter_panel.py "$ALL" "$OUT/analysis/panel" "$OUT/analysis" > "$LOGS/r_panel.txt" 2>&1
note "REPAIR DONE"
