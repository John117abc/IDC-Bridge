#!/bin/bash
# K=1 鑱斿悎瀹為獙锛堜笌鎺у埗鍣ㄨ缁冩椂鐨勯樁娈?A 涓€鑷达級
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
CKPT=/workspace/data/checkpoints/20261119/idc-waymo-v1.0_examples_041839_episode=475.pth
L=/workspace/data/logs
D=/data/data_json/training
W=${W:-12}

echo "===== 1) 瑙勫垝鍣ㄥ崟鏉★紙K=1锛?===="
$P tools/export_planner_plans.py --data-dir $D --k 1 \
    --ckpt /workspace/data/ab60k_B/latest.pt --no-filter --no-prefix \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/planner_k1.npz > "$L/k1_export.log" 2>&1
echo "export exit=$? $(tail -1 $L/k1_export.log)"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --external-candidates /workspace/data/planner_k1.npz \
    --force-path-index 0 --external-now-idx 41 --gif-enabled false > "$L/k1_planner.log" 2>&1
grep -E "Driving Score|Route Completion|NC " "$L/k1_planner.log" | tail -3

echo "===== 2) 浜虹被鍗曟潯锛圞=1锛屾帴鍙ｄ笂闄愶級====="
$P tools/export_planner_plans.py --data-dir $D --k 1 --gt-full --no-prefix --no-filter \
    --files-json /workspace/data/world_files.json \
    --out /workspace/data/gt_k1.npz > "$L/k1_gt_export.log" 2>&1
echo "export exit=$? $(tail -1 $L/k1_gt_export.log)"
$P src/scripts/eval/idc-eval.py --data-dir $D --model-path "$CKPT" \
    --num-worlds $W --seed 42 --external-candidates /workspace/data/gt_k1.npz \
    --force-path-index 0 --external-now-idx 41 --gif-enabled false > "$L/k1_gt.log" 2>&1
grep -E "Driving Score|Route Completion|NC " "$L/k1_gt.log" | tail -3

echo "===== 3) 鍩虹嚎锛堟棤瀹夎锛?===="
grep -E "Driving Score|Route Completion|NC " "$L/joint_base.log" | tail -3

