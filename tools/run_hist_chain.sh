#!/bin/bash
set -u
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
echo "=== 语法检查（用 ast，避免 __pycache__ 权限问题）==="
"$PY" - <<'EOF'
import ast, sys
for f in ('build_vector_dataset.py', 'plant_agent_vec.py'):
    p = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/' + f
    try:
        ast.parse(open(p).read())
        print('OK  ', f)
    except SyntaxError as e:
        print('FAIL', f, e)
EOF
echo "=== 后台重建数据集 + 重训 ==="
L=/mnt/2T_HDD/plant2_vector
nohup bash -c '
  cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
  PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
  T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
  D=/mnt/2T_HDD/plant2_vector
  echo "[chain] 构建数据集 $(date)"
  $PY $T/build_plant2_dataset.py --scenarios all --max-routes 40 \
      --out $D/vector_ds_hist > $D/build_hist.log 2>&1
  echo "[chain] 构建完成 $(date)  文件数: $(ls $D/vector_ds_hist | wc -l)"
  echo "[chain] 训练 WTA 40ep $(date)"
  $PY -u $T/train_plant2_full.py --data $D/vector_ds_hist --out $D/ckpt_hist \
      --epochs 40 --bs 64 > $D/train_hist.log 2>&1
  echo "[chain] WTA done $(date)"
  echo "[chain] 训练 +置信度头 40ep $(date)"
  $PY -u $T/train_plant2_conf.py --data $D/vector_ds_hist --out $D/ckpt_hist_conf \
      --epochs 40 --bs 64 > $D/train_hist_conf.log 2>&1
  echo "[chain] 全部完成 $(date)"
' > $L/chain_hist.log 2>&1 &
sleep 20
cat $L/chain_hist.log
