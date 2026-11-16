#!/usr/bin/env bash
/gpudrive/.venv/bin/python - <<'PY'
import urllib.request
names = [
    "numpy",
    "waymo-open-dataset-tf-2-18-0",
    "waymo-open-dataset-tf-2-16-0",
    "waymo-open-dataset-tf-2-11-0",
    "waymo-open-dataset-tf-2-12-0",
]
for n in names:
    url = f"https://pypi.org/simple/{n}/"
    try:
        with urllib.request.urlopen(url, timeout=10) as f:
            body = f.read(2000).decode("utf-8", "replace")
            hrefs = body.count("href=")
            print(f"{n}: {f.status}, links={hrefs}")
    except Exception as e:
        print(f"{n}: {type(e).__name__} {str(e)[:60]}")
PY
