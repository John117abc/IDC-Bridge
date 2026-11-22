#!/bin/bash
# 联合跑：IDC 闭环 + 上层 Planner 的 K 条模态作为外部候选
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}
PLANNER=${PLANNER:-/workspace/data/ab60k_B/latest.pt}

echo "===== [JOINT] start $(date '+%F %T') =====" | tee "$L/joint_pipeline.log"

echo "[JOINT] 1/3 纯 IDC 基线"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --gif-enabled false > "$L/joint_base.log" 2>&1
echo "  exit=$?" >> "$L/joint_pipeline.log"
grep -E "Driving Score|Route Completion|Collision" "$L/joint_base.log" | tail -3 >> "$L/joint_pipeline.log"

echo "[JOINT] 2/3 导出 Planner 的 K 条模态"
$P tools/export_planner_plans.py --data-dir $D --ckpt "$PLANNER" \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/planner_plans.npz > "$L/joint_export.log" 2>&1
echo "  exit=$?" >> "$L/joint_pipeline.log"
tail -3 "$L/joint_export.log" >> "$L/joint_pipeline.log"

echo "[JOINT] 3/3 联合（外部候选 = planner 模态）"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 \
    --external-candidates /workspace/data/planner_plans.npz \
    --external-now-idx 11 \
    --gif-enabled true --gif-max-worlds 4 --gif-view-mode bird_2d \
    --gif-world-selection first > "$L/joint_ext.log" 2>&1
echo "  exit=$?" >> "$L/joint_pipeline.log"
grep -iE "external|Driving Score|Route Completion|Collision" "$L/joint_ext.log" | tail -5 >> "$L/joint_pipeline.log"
echo "===== [JOINT] DONE $(date '+%F %T') =====" >> "$L/joint_pipeline.log"
cat "$L/joint_pipeline.log"
