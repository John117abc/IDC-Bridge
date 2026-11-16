#!/usr/bin/env bash
P=/gpudrive/.venv/bin/python
echo "=== pip 源配置 ==="
env | grep -i -E "pip_|index" || echo "(无 env)"
$P -m pip config list 2>&1 | tail -3
echo "=== 直连 pypi 查版本 ==="
for v in 2-18-0 2-17-0 2-16-0 2-15-0 2-14-0 2-11-0; do
    out=$($P -m pip index versions --index-url https://pypi.org/simple \
          "waymo-open-dataset-tf-$v" 2>&1 | grep -vE "WARNING" | tail -1)
    echo "tf-$v : $out"
done
