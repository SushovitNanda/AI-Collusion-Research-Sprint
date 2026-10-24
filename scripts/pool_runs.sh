#!/usr/bin/env bash
# Rebuild the derived run folders that are not versioned (see results/README.md):
#   results/extended/runs/eval_pooled  = evaluation batch 1 (results/main) + batch 2 (results/extended)
#   results/model_panel/runs/panel_all = panel seed 1 + seeds 2-3
# evaluate takes one --logs / --dial-logs folder, hence the copies.
set -euo pipefail
cd "$(dirname "$0")/.."
P=results/extended/runs/eval_pooled
rm -rf "$P"; mkdir -p "$P"
cp -r results/main/runs/eval/. "$P"/
cp -r results/extended/runs/eval_b2/. "$P"/
rm -f "$P"/usage.json
echo "eval_pooled: $(ls -d "$P"/*/ | wc -l) runs"
A=results/model_panel/runs/panel_all
rm -rf "$A"; mkdir -p "$A"
cp -r results/model_panel/runs/panel_s1/. "$A"/
cp -r results/model_panel/runs/panel_s23/. "$A"/
rm -f "$A"/usage.json
echo "panel_all: $(ls -d "$A"/*/ | wc -l) runs"
