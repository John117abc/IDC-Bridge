#!/bin/bash
# 修 build_vector_dataset.py 的 id2pos bug，并重跑“两份权重”全流程
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
F=$T/build_vector_dataset.py

echo "=== 1) 打补丁 ==="
"$PY" - "$F" <<'EOF'
import ast, sys
p = sys.argv[1]
src = open(p).read()

# (a) 确保 id2pos 初始化紧跟 frames = []
if 'id2pos = {}' not in src:
    src = src.replace('    frames = []\n', '    frames = []\n    id2pos = {}\n', 1)

# (b) 确保 frame 循环里填 id2pos
fill = ("        frames.append((f, boxes, ego[0]))\n"
        "        for _b in boxes:\n"
        "            _id = _b.get('id')\n"
        "            if _id is None:\n"
        "                continue\n"
        "            id2pos.setdefault(_id, {})[f] = (_b['wx'], _b['wy'])\n")
if 'id2pos.setdefault' not in src:
    src = src.replace('        frames.append((f, boxes, ego[0]))\n', fill, 1)

open(p, 'w').write(src)
ast.parse(src)
print('PATCH_OK  id2pos_init=%s  id2pos_fill=%s'
      % ('id2pos = {}' in src, 'id2pos.setdefault' in src))
EOF
[ $? -ne 0 ] && { echo "补丁失败，停止"; exit 1; }

echo "=== 2) 冒烟：单条 route 构建 ==="
R=$("$PY" - <<'EOF'
import glob
rs = sorted(glob.glob('/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data/HighwayCutIn/Town12*'))
print(rs[0] if rs else '')
EOF
)
echo "route=$R"
"$PY" "$T/build_vector_dataset.py" --route "$R" --map "$D/maps/map_Town12.npz" \
    --out "$D/smoke_hist.npz" 2>&1 | tail -8
[ -f "$D/smoke_hist.npz" ] || { echo "冒烟失败，停止"; exit 1; }

echo "=== 3) 重启全流程（两份权重）$(date) ==="
nohup bash -c '
  PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
  T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
  D=/mnt/2T_HDD/plant2_vector
  echo "[chain] 构建 $(date)"
  $PY $T/build_plant2_dataset.py --scenarios all --max-routes 40 --out $D/vector_ds_hist > $D/build_hist.log 2>&1
  echo "[chain] 构建完成 $(ls $D/vector_ds_hist | wc -l) 个文件 $(date)"
  echo "[chain] 训练A: 无置信度头（给 IDC 选路）$(date)"
  $PY -u $T/train_plant2_full.py --data $D/vector_ds_hist --out $D/ckpt_idcsel --epochs 40 --bs 64 > $D/train_idcsel.log 2>&1
  cp -f $D/ckpt_idcsel/best.pt $D/plant2_idcsel.pt
  echo "[chain] A 完成 $(date)"
  echo "[chain] 训练B: 有置信度头 $(date)"
  $PY -u $T/train_plant2_conf.py --data $D/vector_ds_hist --out $D/ckpt_conf_hist --epochs 40 --bs 64 > $D/train_conf.log 2>&1
  cp -f $D/ckpt_conf_hist/best.pt $D/plant2_conf.pt
  echo "[chain] B 完成 $(date)"
' > $D/chain_hist.log 2>&1 &
sleep 25
cat $D/chain_hist.log
