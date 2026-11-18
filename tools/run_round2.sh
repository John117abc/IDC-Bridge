#!/usr/bin/env bash
# Round-2 通宵流水线：重建(因果) → 重训 → 2×2 网格 → 汇总
set -u
cd /gpudrive/baselines/IDC-Bridge || exit 1
P=/gpudrive/.venv/bin/python
L=/workspace/data/logs
D=/workspace/data/head_causal2
mkdir -p "$L"
SUM=$L/round2_summary.txt
export PYTHONUNBUFFERED=1

echo "===== ROUND2 start $(date '+%F %T') =====" > "$SUM"

# 1) 重建数据集（自车 + 周车速度均因果）
$P tools/build_head_dataset.py --data-dir /data/data_json/training \
    --n-scenes 60000 --seed 0 --out "$D" --k 20 > "$L/build_causal2.log" 2>&1
echo "[1] build exit=$? $(date '+%T')" >> "$SUM"
tail -3 "$L/build_causal2.log" >> "$SUM"
if [ ! -f "$D/train.npz" ]; then
    echo "FATAL: no train.npz" >> "$SUM"; exit 1
fi

# 2) 重训规划头
$P tools/train_head.py --data "$D" --epochs 200 --bs 128 --lr 2e-4 \
    --patience 8 --out "$D/head_best.pt" > "$L/train_causal2.log" 2>&1
echo "[2] train exit=$? $(date '+%T')" >> "$SUM"
tail -3 "$L/train_causal2.log" >> "$SUM"
if [ ! -f "$D/head_best.pt" ]; then
    echo "FATAL: no head_best.pt" >> "$SUM"; exit 1
fi

# 3) 2×2 网格实验（判据对齐 + 统一起点）
$P tools/exp_grid.py --data "$D" --model "$D/head_best.pt" \
    --data-dir /data/data_json/training \
    --files-json /workspace/data/holdout_files.json \
    --n-scenes 2000 --out /workspace/data/grid2 > "$L/grid2.log" 2>&1
echo "[3] grid exit=$? $(date '+%T')" >> "$SUM"
sed -n '/场景级/,$p' "$L/grid2.log" >> "$SUM"

echo "===== ROUND2 DONE $(date '+%F %T') =====" >> "$SUM"
