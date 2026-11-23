#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)
CK=${CK:-$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261122/*.pth | head -1)}
echo "权重=$CK"

export VEC_CTRL=idc
export VEC_ROUTE_ONLY=${VEC_ROUTE_ONLY:-1}
export VEC_IDC_CKPT="$CK"
export VEC_DEBUG=1                     # 记录 dbg（route/modes/agents/控制）
export VEC_STUCK_SEC=${VEC_STUCK_SEC:-8}
export RUN_NAME=vis_$(date +%H%M%S)
echo "RUN=$RUN_NAME route_only=$VEC_ROUTE_ONLY stuck=${VEC_STUCK_SEC}s"
rm -f /mnt/2T_HDD/plant2_vector/dbg_*.npz
bash $ROOT/tools/run_b2d_smoke.sh > /tmp/carla_vis.log 2>&1
echo "exit=$?"
bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 | grep -E 'route Route|infractions' | head -4
echo "=== 卡死终止记录 ==="
grep -h 'STUCK' /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME/*.log 2>/dev/null | head -3
echo "=== dbg 文件 ==="
ls -la /mnt/2T_HDD/plant2_vector/dbg_*.npz 2>/dev/null | head -3
