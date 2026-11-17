#!/usr/bin/env bash
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261115/idc-waymo-v1.0_examples_201537_episode=3000.pth
L=/workspace/data/logs
D=/data/data_json/training

echo "===== [EXT] start $(date '+%F %T') =====" > "$L/ext_pipeline.log"

# 1) dump 本次实际用到的场景文件（同时这就是地图候选基线）
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --gif-enabled false > "$L/ext_step1.log" 2>&1
echo "[EXT] step1(基线+dump) exit=$?" >> "$L/ext_pipeline.log"
grep -E "Driving Score|Route Completion|NC " "$L/ext_step1.log" | tail -3 >> "$L/ext_pipeline.log"

# 2) 按 dump 的文件名导出规划头 top-5
$P tools/export_head_plans.py --data-dir $D \
    --head-data /workspace/data/head_dd/medoid \
    --model /workspace/data/head_dd/medoid/head_best.pt \
    --num-worlds 12 --seed 42 --files-json /workspace/data/world_files.json \
    --out /workspace/data/head_plans2.npz > "$L/ext_step2.log" 2>&1
echo "[EXT] step2(导出) exit=$?" >> "$L/ext_pipeline.log"
tail -2 "$L/ext_step2.log" >> "$L/ext_pipeline.log"

# 3) 用外部候选跑 GEP
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds 12 --seed 42 --external-candidates /workspace/data/head_plans2.npz \
    --gif-enabled true --gif-max-worlds 3 --gif-view-mode bird_2d \
    --gif-world-selection first > "$L/ext_step3.log" 2>&1
echo "[EXT] step3(外部候选) exit=$?" >> "$L/ext_pipeline.log"
grep -i external "$L/ext_step3.log" | head -2 >> "$L/ext_pipeline.log"
grep -E "Driving Score|Route Completion|NC " "$L/ext_step3.log" | tail -3 >> "$L/ext_pipeline.log"
echo "===== [EXT] DONE $(date '+%F %T') =====" >> "$L/ext_pipeline.log"
