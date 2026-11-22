#!/bin/bash
# 等全量构建结束，然后跑 3 epoch 训练
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector/vector_ds_full
B=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data
L=/mnt/2T_HDD/plant2_vector

echo "场景总数: $(ls -d $B/*/ | wc -l)"
echo "等待构建进程结束..."
for i in $(seq 1 240); do
  n=$(pgrep -f build_plant2_dataset | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 30
done
echo "构建结束 $(date)"
echo "npz 文件数: $(ls $D | wc -l)"
du -sh $D

echo "=== 3 epoch 训练开始 $(date) ==="
"$PY" "$T/train_plant2_small.py" --data "$D" --epochs 3 > "$L/train_full.log" 2>&1
echo "exit=$?"
tail -8 "$L/train_full.log"
echo "=== 完成 $(date) ==="
