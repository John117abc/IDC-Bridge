#!/bin/bash
set -u
docker exec idc_hist bash -lc 'cd /gpudrive/baselines/IDC-Bridge && timeout 600 /gpudrive/.venv/bin/python tools/collect_v2_pairs.py --ckpt /workspace/data/plant2_ckpt/best.pt --out /workspace/data/v2_smoke.npz --worlds 1 --steps 3 --k 1 --data_root /data/data_json/training > /tmp/collect.log 2>&1; echo "exit=$?"'
echo "=== 关键行 ==="
docker exec idc_hist bash -lc 'grep -E "ScenePack|RT\]|collect\]|Error|Traceback" /tmp/collect.log | head -20'
