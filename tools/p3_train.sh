#!/bin/bash
# P3 夜间训练：occ 监督（aux 重建），从 ckpt_occpos 热启动
set -u
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
D=/mnt/2T_HDD/plant2_vector
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
LOG=$D/train_occaux.log

cd $R/tools
export CUDA_VISIBLE_DEVICES=1
export VEC_INIT_CKPT=$D/ckpt_occpos/best.pt
nohup $PY train_plant2_occ.py \
  --occ-aux --lam-occ 0.1 \
  --data $D/vector_ds_occ --out $D/ckpt_occaux \
  --epochs 10 --bs 48 --lr 2e-4 --lam-ce 0.5 --patience 3 \
  > $LOG 2>&1 &
echo "launched pid=$!"
sleep 60
echo "=== 前 60 秒 ==="
tail -5 $LOG
