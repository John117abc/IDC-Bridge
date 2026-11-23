#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
pkill -f run_hist_chain 2>/dev/null
pkill -f build_plant2_dataset 2>/dev/null
pkill -f train_plant2_full 2>/dev/null
pkill -f train_plant2_conf 2>/dev/null
sleep 2
echo "残留: $(pgrep -f 'build_plant2_dataset|train_plant2' | wc -l)"

$PY - "$T" <<'EOF'
import ast, sys, os
T = sys.argv[1]
for f in ('plant_agent_vec.py', 'build_vector_dataset.py'):
    p = os.path.join(T, f)
    try:
        ast.parse(open(p).read()); print('AST_OK  ', f)
    except SyntaxError as e:
        print('AST_FAIL', f, e); raise SystemExit(1)
EOF

if [ $? -ne 0 ]; then echo "语法错误，不启动训练"; exit 1; fi

echo "重启链式任务（历史版）$(date)"
nohup bash -c '
  PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
  T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
  D=/mnt/2T_HDD/plant2_vector
  echo "[chain] 构建 $(date)"
  $PY $T/build_plant2_dataset.py --scenarios all --max-routes 40 --out $D/vector_ds_hist > $D/build_hist.log 2>&1
  echo "[chain] 构建完成 文件数 $(ls $D/vector_ds_hist | wc -l) $(date)"
  $PY -u $T/train_plant2_full.py --data $D/vector_ds_hist --out $D/ckpt_hist --epochs 40 --bs 64 > $D/train_hist.log 2>&1
  echo "[chain] WTA 完成 $(date)"
  $PY -u $T/train_plant2_conf.py --data $D/vector_ds_hist --out $D/ckpt_hist_conf --epochs 40 --bs 64 > $D/train_hist_conf.log 2>&1
  echo "[chain] 全部完成 $(date)"
' > $D/chain_hist.log 2>&1 &
sleep 15
cat $D/chain_hist.log
