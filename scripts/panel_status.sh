#!/usr/bin/env bash
# One status block for the OpenRouter panel: pipeline progress + OpenRouter spend + judge tokens.
cd "$(dirname "$0")/.."
set -a; source .env; set +a
python scripts/panel_progress.py results/model_panel
python - <<'EOF'
import json, os, glob, urllib.request
try:
    req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key",
                                 headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]})
    d = json.load(urllib.request.urlopen(req, timeout=20))["data"]
    used = float(d.get("usage", 0))
    # user reported an $8 balance at ~$1.4 used (24 Oct 12:50): balance ~= 9.4 - usage
    bal = 9.4 - used
    print(f"  openrouter spend: ${used:.2f} used; estimated account balance ${bal:.2f}"
          + ("  WARNING: balance low, stop launching new runs" if bal < 3 else ""))
except Exception as e:
    print(f"  openrouter spend: unavailable ({type(e).__name__})")
n = 0
for f in glob.glob("results/model_panel/runs/panel_s*/*/monitor/glm__*.json") + glob.glob("results/model_panel/runs/panel_s*/*/monitor/claims__*.json"):
    try:
        u = json.load(open(f, encoding="utf8")).get("usage") or {}
        n += int(u.get("input_tokens", 0)) + int(u.get("output_tokens", 0))
    except Exception:
        pass
print(f"  modal judge tokens so far: {n:,} (~${n / 1e6 * 3.6:.2f} at the measured blended rate)")
EOF
