#!/bin/bash
set -u
ROOT=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
CK=$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261123/*.pth 2>/dev/null | head -1)
CK=${CK:-$(ls -t /mnt/2T_HDD/WaymoDatabas/checkpoints/20261122/*.pth | head -1)}
echo "新权重 = $CK"

run_case() {
  local tag=$1; local route_only=$2; shift 2
  export VEC_CTRL=idc
  export VEC_ROUTE_ONLY=$route_only
  export VEC_IDC_CKPT="$CK"
  export RUN_NAME="${tag}_$(date +%H%M%S)"
  echo "--- $tag (route_only=$route_only) ---"
  bash $ROOT/tools/run_b2d_smoke.sh > /tmp/carla_$tag.log 2>&1
  bash $ROOT/tools/show_results.sh /mnt/2T_HDD/plant2_vector/closedloop/$RUN_NAME 2>&1 \
    | grep -E 'route Route|infractions' | head -3
}

echo "=== A) route 参考 + IDC（新权重）==="
run_case idc_route_new 1
echo "=== B) 规划器 + IDC（新权重）==="
run_case idc_planner_new 0
