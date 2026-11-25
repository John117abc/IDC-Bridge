#!/bin/bash
# 链式：等栅格重生成 → 重建全量数据 → 自动训练
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
LOG=$D/chain_occ.log
ROOT=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data

echo "[$(date +%H:%M:%S)] 链式任务开始，等待栅格重生成" >> "$LOG"

# 1) 等 make_occupancy 结束
while pgrep -f "make_occupancy.py" > /dev/null 2>&1; do sleep 60; done
echo "[$(date +%H:%M:%S)] 栅格重生成结束" >> "$LOG"

# 2) 抽查栅格：行人通道应有非零
$PY - >> "$LOG" 2>&1 <<'EOF'
import glob, os
import numpy as np
fs = glob.glob('/mnt/2T_HDD/plant2_vector/occupancy/PedestrianCrossing__*.npz')
tot = 0
for f in fs[:16]:
    o = np.load(f)['occ']
    tot += int(o[:, 1].sum())
print(f'行人通道抽检：{len(fs[:16])} 个 PedestrianCrossing 文件，行人格总数 {tot}')
EOF

# 3) 旧数据改名（保留，不删除）
if [ -d "$D/vector_ds_occ" ]; then
  mv "$D/vector_ds_occ" "$D/vector_ds_occ_stale_$(date +%H%M)" 2>/dev/null
fi
mkdir -p "$D/vector_ds_occ"

# 4) 全量重建
echo "[$(date +%H:%M:%S)] 开始重建全量数据" >> "$LOG"
for k in 0 1 2 3; do
  [ -f /tmp/sc_part_$k.txt ] || continue
  LIST=$(paste -sd, /tmp/sc_part_$k.txt)
  cd $T
  SKIP_DONE=0 PYTHONUNBUFFERED=1 nohup $PY build_plant2_dataset.py \
    --data-root $ROOT --out "$D/vector_ds_occ" --scenarios "$LIST" --max-routes 0 \
    >> "$D/rebuild_occ2.log" 2>&1 &
done

# 5) 等 39 个文件
while [ "$(ls $D/vector_ds_occ/*.npz 2>/dev/null | wc -l)" -lt 39 ]; do
  sleep 120
  N=$(ls $D/vector_ds_occ/*.npz 2>/dev/null | wc -l)
  echo "[$(date +%H:%M:%S)] 重建进度 $N/39" >> "$LOG"
  if ! pgrep -f "build_plant2_dataset" > /dev/null 2>&1; then
    echo "[$(date +%H:%M:%S)] !! worker 全退出，当前 $N/39，重启一次" >> "$LOG"
    for k in 0 1 2 3; do
      [ -f /tmp/sc_part_$k.txt ] || continue
      LIST=$(paste -sd, /tmp/sc_part_$k.txt)
      cd $T
      SKIP_DONE=1 PYTHONUNBUFFERED=1 nohup $PY build_plant2_dataset.py \
        --data-root $ROOT --out "$D/vector_ds_occ" --scenarios "$LIST" \
        --max-routes 0 >> "$D/rebuild_occ2.log" 2>&1 &
    done
  fi
done

# 6) 训练
echo "[$(date +%H:%M:%S)] 重建完成，启动训练" >> "$LOG"
cd $T
nohup $PY train_plant2_occ.py --data "$D/vector_ds_occ" \
  --out "$D/ckpt_occ" --epochs 20 --bs 48 --lr 3e-4 \
  > "$D/train_occ.log" 2>&1 &
echo "[$(date +%H:%M:%S)] 训练已启动 pid=$!" >> "$LOG"
