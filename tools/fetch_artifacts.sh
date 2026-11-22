#!/bin/bash
# 获取 / 重建 PlanT2 向量规划器的产物（数据集 + 权重）
#
# 用法:
#   bash tools/fetch_artifacts.sh ckpt    # 只取权重（仓库内 assets/ 或 $CKPT_URL）
#   bash tools/fetch_artifacts.sh maps    # 只导出 CARLA 地图折线
#   bash tools/fetch_artifacts.sh data    # 只构建向量数据集
#   bash tools/fetch_artifacts.sh train   # 只训练（先 WTA 40ep，再 +置信度头 40ep）
#   bash tools/fetch_artifacts.sh all     # 全流程（maps → data → train）
#   bash tools/fetch_artifacts.sh check   # 只校验现有产物
#
# 环境变量（都有默认值，按需覆盖）:
#   ARTIFACT_DIR  产物根目录       默认 /mnt/2T_HDD/plant2_vector
#   PY            python 解释器    默认 bench2drive_b2d 环境（需 torch+scipy+carla）
#   PLANT2_DATA   PlanT2 数据集根  默认 /mnt/2T_HDD/PlanT2.0/.../PlanT_2_dataset/data
#   CKPT_URL      权重下载地址（可选；给了就优先从它下载，而不是用仓库内文件）
#
# 说明: 数据与权重不进 git（除 assets/ 下这份 14 MB 权重），产物全部落在 $ARTIFACT_DIR。
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${ARTIFACT_DIR:-/mnt/2T_HDD/plant2_vector}
PY=${PY:-/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python}
PLANT2_DATA=${PLANT2_DATA:-/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data}
CKPT_SHA=c53dd6278f701dd3b873e2fff4ca9ebf954b4f3c9280b8e102e5bce19aa56611
CKPT_OUT=$BASE/ckpt_conf/best.pt
REPO_CKPT=$ROOT/assets/plant2_ckpt_conf_best.pt

# CARLA .xodr 的两个来源（Town01-10 在 download/AdditionalMaps，Town11-15 在 CARLA_0.9.16）
XODR_A=/mnt/2T_HDD/download/AdditionalMaps_0.9.16/CarlaUE4/Content/Carla/Maps/OpenDrive
XODR_B=/mnt/2T_HDD/CARLA_0.9.16/CarlaUE4/Content/Carla/Maps

log() { echo "[fetch] $*"; }
die() { echo "[fetch][error] $*" >&2; exit 1; }

mkdir -p "$BASE"

fetch_ckpt() {
  mkdir -p "$(dirname "$CKPT_OUT")"
  if [ -n "${CKPT_URL:-}" ]; then
    log "从 CKPT_URL 下载权重"
    curl -L --fail -o "$CKPT_OUT" "$CKPT_URL"
  elif [ -f "$REPO_CKPT" ]; then
    log "使用仓库内权重 assets/plant2_ckpt_conf_best.pt"
    cp -f "$REPO_CKPT" "$CKPT_OUT"
  else
    die "既没有 CKPT_URL，也没有 $REPO_CKPT；请先 bash tools/fetch_artifacts.sh train"
  fi
  got=$(sha256sum "$CKPT_OUT" | awk '{print $1}')
  if [ "$got" != "$CKPT_SHA" ]; then
    log "警告: sha256 不匹配（期望 $CKPT_SHA，实际 $got）"
  else
    log "权重校验通过: $CKPT_OUT"
  fi
}

build_maps() {
  log "导出 CARLA 地图折线 → $BASE/maps"
  [ -d "$XODR_A" ] || die "缺少 .xodr 目录: $XODR_A"
  "$PY" "$ROOT/tools/export_map_polylines.py" \
       --xodr-dirs "$XODR_A,$XODR_B" --out "$BASE/maps"
  ls "$BASE/maps" | head -5
}

build_data() {
  log "构建向量数据集 → $BASE/vector_ds_full"
  [ -d "$PLANT2_DATA" ] || die "缺少 PlanT2 数据集: $PLANT2_DATA"
  [ -d "$BASE/maps" ] || build_maps
  "$PY" "$ROOT/tools/build_plant2_dataset.py" \
       --data-root "$PLANT2_DATA" --maps "$BASE/maps" \
       --scenarios all --max-routes 40 --out "$BASE/vector_ds_full"
  ls "$BASE/vector_ds_full" | wc -l
}

train_all() {
  log "训练 1/2: 纯 WTA 40 epoch"
  "$PY" "$ROOT/tools/train_plant2_full.py" --data "$BASE/vector_ds_full" \
       --out "$BASE/ckpt" --epochs 40 --bs 64
  log "训练 2/2: + 置信度头 40 epoch"
  "$PY" "$ROOT/tools/train_plant2_conf.py" --data "$BASE/vector_ds_full" \
       --out "$BASE/ckpt_conf" --epochs 40 --bs 64
  cp -f "$BASE/ckpt_conf/best.pt" "$CKPT_OUT"
  log "训练完成: $CKPT_OUT"
}

check_all() {
  log "检查产物"
  for f in "$BASE/maps" "$BASE/vector_ds_full" "$CKPT_OUT"; do
    if [ -e "$f" ]; then echo "  OK   $f"; else echo "  MISS $f"; fi
  done
  [ -f "$CKPT_OUT" ] && sha256sum "$CKPT_OUT"
  [ -d "$BASE/vector_ds_full" ] && du -sh "$BASE/vector_ds_full"
}

case "${1:-check}" in
  ckpt)  fetch_ckpt ;;
  maps)  build_maps ;;
  data)  build_data ;;
  train) train_all ;;
  all)   build_maps; build_data; train_all; fetch_ckpt ;;
  check) check_all ;;
  *) die "未知参数: $1（可用: ckpt/maps/data/train/all/check）" ;;
esac
