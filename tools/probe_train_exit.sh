#!/bin/bash
set -u
CR=/gpudrive/baselines/IDC-Bridge
echo "=== sys.exit / except 包裹 ==="
docker exec idc_hist bash -lc "grep -nE 'sys.exit|except Exception|KeyboardInterrupt|return$' $CR/src/scripts/train/idc-train.py | head -20"
echo
echo "=== main 函数签名与结尾 ==="
docker exec idc_hist bash -lc "grep -nE '^def |^if __name__' $CR/src/scripts/train/idc-train.py | head -10"
echo
echo "=== 末尾 20 行 ==="
docker exec idc_hist bash -lc "tail -20 $CR/src/scripts/train/idc-train.py"
