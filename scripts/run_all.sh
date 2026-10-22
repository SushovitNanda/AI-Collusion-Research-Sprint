#!/usr/bin/env bash
# Full solo pipeline. Run the pilot first and check the go/no-go criteria in README.
set -euo pipefail
AGENT=${AGENT:-glm}
JUDGE=${JUDGE:-qwen}
python -m collusion_dial.grid --plan solo --model "$AGENT" --yes
python -m collusion_dial.monitor --logs logs/solo --judge-model "$JUDGE"
python -m collusion_dial.monitor --logs logs/solo --judge-model "$JUDGE" --tags
python -m collusion_dial.evaluate --logs logs/solo --out results/solo --tagger "$JUDGE"
