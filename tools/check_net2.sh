#!/usr/bin/env bash
echo "--- proxy env ---"
env | grep -i proxy || echo "(none)"
echo "--- dns ---"
getent hosts pypi.org 2>&1 | head -2 || true
echo "--- https test ---"
/gpudrive/.venv/bin/python - <<'PY'
import urllib.request
for url in ("https://pypi.org/simple/", "https://files.pythonhosted.org/"):
    try:
        with urllib.request.urlopen(url, timeout=8) as f:
            print(url, "->", f.status)
    except Exception as e:
        print(url, "->", type(e).__name__, str(e)[:80])
PY
