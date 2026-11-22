#!/bin/bash
# 验证：target_pos 重锚定 是否是"人类轨迹也掉分"的根因
# 1) 单帧(frames=0) + GT 参考，走 --replan-candidates（带 target_pos 重锚定）
# 2) 同一份参考走 --external-candidates（不带重锚定）作对照
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

echo "=== 1) 导出单帧 GT 参考 ==="
$P tools/export_replan.py --data-dir $D --gt --frames 0 --k 1 \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/replan_gt0.npz > "$L/replan_gt0_export.log" 2>&1
echo "export exit=$? $(tail -1 $L/replan_gt0_export.log)"

echo "=== 2) replan 通路（带 target_pos 重锚定）==="
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --replan-candidates /workspace/data/replan_gt0.npz \
    --force-path-index 0 --gif-enabled false > "$L/replan_gt0.log" 2>&1
grep -E "REPLAN|Driving Score|Route Completion|NC " "$L/replan_gt0.log" | tail -5

echo "=== 3) 对照：external 通路（无重锚定）==="
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --external-candidates /workspace/data/replan_gt0.npz \
    --external-now-idx 41 --force-path-index 0 --gif-enabled false > "$L/ext_gt0.log" 2>&1
grep -E "Driving Score|Route Completion|NC " "$L/ext_gt0.log" | tail -3
