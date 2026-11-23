#!/bin/bash
set -u
echo "=== 候选目录文件统计 ==="
docker exec idc_hist bash -lc '
for d in /data/data /data/data_json /gpudrive/data/processed /workspace/data/ab60k; do
  n=$(ls $d 2>/dev/null | wc -l)
  ext=$(ls $d 2>/dev/null | sed "s/.*\.//" | sort -u | head -3 | tr "\n" " ")
  echo "$d : $n 个, 扩展名: $ext"
done'
echo "=== 自动挑选并跑冒烟 ==="
docker exec idc_hist bash -lc '
cd /gpudrive/baselines/IDC-Bridge
DATA=$(for d in /data/data /gpudrive/data/processed /workspace/data/ab60k; do
  n=$(ls $d 2>/dev/null | grep -c "\.json$")
  [ "$n" -gt 50 ] && echo "$d" && break
done)
echo "选用数据目录: $DATA"
timeout 900 /gpudrive/.venv/bin/python tools/collect_v2_pairs.py \
  --ckpt /workspace/data/plant2_ckpt/best.pt \
  --out /workspace/data/v2_smoke.npz \
  --worlds 2 --steps 12 --k 1 --data_root "$DATA" 2>&1 | tail -18
'
