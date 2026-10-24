#!/usr/bin/env bash
# Full end-to-end run (v0.3 configuration, validated in pilots 1-3; see README, 'Pilot history').
# Order matters: the evaluation set freezes thresholds and the hybrid before any dial run is scored.
set -euo pipefail
AGENT=${AGENT:-qwen}
JUDGE=${JUDGE:-glm}
EXTRACTOR=${EXTRACTOR:-qwen}
DIAL_SEEDS=${DIAL_SEEDS:-1,2}

# 1. evaluation set (24 runs)
python -m collusion_dial.grid --plan eval --model "$AGENT" --parallel-runs 6 --yes
python -m collusion_dial.monitor --logs logs/eval --judge-model "$JUDGE" --extractor-model "$EXTRACTOR" --concurrency 8
python -m collusion_dial.evaluate --logs logs/eval --out outputs/eval
python scripts/go_no_go.py logs/eval outputs/eval

# 2. dial, rule arm (fixed lambda only) and no-rule arm (fixed + sweeps)
python -m collusion_dial.grid --plan dial --model "$AGENT" --seeds "$DIAL_SEEDS" --fixed-only --yes
python -m collusion_dial.grid --plan dial_norule --model "$AGENT" --seeds "$DIAL_SEEDS" --yes
python -m collusion_dial.monitor --logs logs/dial        --judge-model "$JUDGE" --extractor-model "$EXTRACTOR"
python -m collusion_dial.monitor --logs logs/dial_norule --judge-model "$JUDGE" --extractor-model "$EXTRACTOR"
python -m collusion_dial.monitor --logs logs/dial_norule --judge-model "$JUDGE" --extractor-model "$EXTRACTOR" --claims-only --include-sweeps
python -m collusion_dial.evaluate --logs logs/eval --dial-logs logs/dial        --out outputs/main_rule
python -m collusion_dial.evaluate --logs logs/eval --dial-logs logs/dial_norule --out outputs/main_norule

# 3. optional: judge with a stake (masking), label-1 runs only matter
python -m collusion_dial.monitor --logs logs/eval --judge-model "$JUDGE" --extractor-model "$EXTRACTOR" --views saydo,structured --stake 0.5
python -m collusion_dial.monitor --logs logs/eval --judge-model "$JUDGE" --extractor-model "$EXTRACTOR" --views saydo,structured --stake 1.0
python -m collusion_dial.evaluate --logs logs/eval --out outputs/eval
