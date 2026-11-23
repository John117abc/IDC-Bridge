#!/bin/bash
# 快速测试：单条路线 + 硬超时（默认 240 s）+ 卡死 6 s 即终止
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
SRC=${ROUTES_FILE:-/mnt/2T_HDD/plant2_vector/other_towns.xml}
IDX=${ROUTE_IDX:-0}          # 取第几条路线
ONE=/mnt/2T_HDD/plant2_vector/one_route.xml
LIMIT=${LIMIT:-240}

$PY - "$SRC" "$ONE" "$IDX" <<'EOF'
import sys, xml.etree.ElementTree as ET
src, out, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
root = ET.parse(src).getroot()
rs = root.findall('route')
sel = rs[idx]
new = ET.Element('routes'); new.append(sel)
ET.ElementTree(new).write(out)
print('单路线:', sel.get('id'), sel.get('town'))
EOF

export ROUTES_FILE=$ONE
export VEC_CTRL=${VEC_CTRL:-idc}
export VEC_ROUTE_ONLY=${VEC_ROUTE_ONLY:-1}
export VEC_IDC_CKPT=${VEC_IDC_CKPT:-/mnt/2T_HDD/WaymoDatabas/checkpoints/bc/idc_bc.pth}
export VEC_STUCK_SEC=${VEC_STUCK_SEC:-6}
export VEC_DEBUG=${VEC_DEBUG:-1}
export RUN_NAME=fast_$(date +%H%M%S)
echo "RUN=$RUN_NAME 超时=${LIMIT}s 卡死=${VEC_STUCK_SEC}s"
timeout $LIMIT bash $ROOT/tools/run_b2d_smoke.sh > /tmp/fast_test.log 2>&1
echo "exit=$? (124=超时被杀)"
D=/mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME
bash $ROOT/tools/show_results.sh $D 2>&1 | grep -E 'route Route|infractions' | head -3
echo "=== 诊断 ==="
grep -hE '\[PID\]|\[IDC\]|IDCdbg' $D/*.log 2>/dev/null | head -4
grep -hE '\[PID\]|\[IDC\]|IDCdbg' $D/*.log 2>/dev/null | tail -3
