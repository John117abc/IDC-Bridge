#!/bin/bash
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
CG=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage
D=/mnt/2T_HDD/plant2_vector

for p in $(pgrep -f "run_panel_gif.sh" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f "run_b2d_smoke.sh" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f "CarlaUE4-Linux-Shipping" 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
sleep 3

$PY $T/patch_dbg_save_occ.py
cp -f $T/plant_agent_vec.py $CG/plant_agent_vec.py
$PY - <<'EOF'
import ast, io
p = '/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/plant_agent_vec.py'
ast.parse(io.open(p, encoding='utf-8').read()); print('syntax OK')
EOF

# 用新脚本跑两条路线
rm -rf $D/video/* 2>/dev/null; rm -f $D/dbg_*.npz
export VEC_CTRL=idc VEC_ROUTE_ONLY=0 VEC_REF_FROM=route
export VEC_CKPT=$D/ckpt_occ/best.pt
export VEC_IDC_CKPT=/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_60ep.pth
export VEC_STUCK_SEC=25 VEC_DEBUG=1 VEC_RECORD=1 VEC_SAFETY=1
export VEC_IDC_VMAX=14 VEC_IDC_VMIN=5.5
export VEC_LAT_MODE=pp VEC_PP_MAX=0.75 VEC_PP_RATE=0.35
export ROUTES_FILE=$D/two_routes_sub.xml
export RUN_NAME=panel2_$(date +%H%M%S)
export RUN_TIMEOUT=420
bash $T/run_b2d_smoke.sh > /tmp/panel2_run.log 2>&1
bash $T/show_results.sh $D/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route' | head -3
DBG=$(ls $D/dbg_1.npz 2>/dev/null || ls $D/dbg_0.npz 2>/dev/null)
echo "dbg=$DBG"
$PY $T/compose_panel_gif2.py --video $D/video --dbg "$DBG" --out $D/panel_v2.gif --every 2 --max 150 2>&1 | tail -3
