#!/bin/bash
# P3 自动流水线：判有效 → 加训 → 判损失稳定 → 存模型
#
# 判据：
#   Δ = probe 的全局 max|Δ|（障碍交换后轨迹变化）
#   基线（未加 occ 监督）= 8e-6
#   Δ ≥ 1e-3        → 有效，继续加训
#   1e-4 ≤ Δ < 1e-3 → 部分有效，再给一轮
#   Δ < 1e-4        → 无效，停止并写结论
# 停止条件（任一）：
#   1) Δ < 1e-4
#   2) 连续两轮 val minFDE 改善 < 0.5%
#   3) 已到 stage6（合计 ≤ ~60 epoch）
set -u
D=/mnt/2T_HDD/plant2_vector
R=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
LOG=$D/p3_pipeline.log
STATUS=$D/p3_pipeline_status.txt
PROBE=$D/probe_auto.txt

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG" > /dev/null; }
say() { echo "$*" | tee -a "$LOG" "$STATUS"; }

wait_stage2() {
  until grep -q "stage2 结束" $D/train_stage2.log 2>/dev/null; do sleep 60; done
}

wait_no_train() {
  while pgrep -f "train_plant2_occ.py" > /dev/null 2>&1; do sleep 60; done
}

pick_ckpt() {
  for c in $D/ckpt_occposaux6/best.pt $D/ckpt_occposaux5/best.pt \
           $D/ckpt_occposaux4/best.pt $D/ckpt_occposaux3/best.pt \
           $D/ckpt_occposaux2/best.pt $D/ckpt_occposaux/best.pt; do
    [ -f "$c" ] && { echo "$c"; return; }
  done
  echo ""
}

run_probe() {
  local ck=$1
  cd $R/tools || return
  CUDA_VISIBLE_DEVICES=0 $PY probe_occ_sensitivity.py --ckpt "$ck" \
      --data $D/vector_ds_occ/ConstructionObstacle.npz --out $PROBE >> "$LOG" 2>&1
  grep -oE "全局 max\|Δ\| = [0-9.eE+-]+" "$LOG" | tail -1 | awk '{print $NF}'
}

say "==== P3 自动流水线启动 $(date) ===="
wait_stage2
say "stage2 完成标记已出现"

declare -A prevfde
stage=2
last_delta=0
while [ $stage -le 6 ]; do
  wait_no_train
  ck=$(pick_ckpt)
  if [ -z "$ck" ]; then say "找不到 ckpt，停止"; break; fi
  say "stage$((stage-1)) 结束，当前 ckpt=$ck"

  # 保存一份该阶段模型
  mkdir -p $D/ckpt_saved
  cp -f "$ck" $D/ckpt_saved/stage$((stage-1))_best.pt 2>/dev/null

  delta=$(run_probe "$ck")
  say "probe Δ=$delta（基线 8e-6，有效阈值 1e-3）"
  last_delta=$delta

  # 读该阶段 val minFDE
  fde=$(grep -h "训练完成 best minFDE" $D/train_occposaux.log \
        $D/train_stage2.log $D/train_stage3.log $D/train_stage4.log \
        $D/train_stage5.log $D/train_stage6.log 2>/dev/null \
        | tail -1 | grep -oE "[0-9.]+$")
  say "当前 val minFDE=$fde"

  eff=$(python3 - "$delta" <<'EOF'
import sys
try:
    d = float(sys.argv[1])
except Exception:
    print('unknown'); raise SystemExit
print('effective' if d >= 1e-3 else ('partial' if d >= 1e-4 else 'ineffective'))
EOF
)
  say "判定=$eff"

  if [ "$eff" = "unknown" ] || [ -z "$delta" ]; then
    say "→ probe 解析失败（delta='$delta'），停止并保留现场"
    break
  fi

  if [ "$eff" = "ineffective" ]; then
    say "→ 监督无效（Δ<1e-4），停止加训。结论：需换更强监督（逐 patch 分类/对比学习）"
    break
  fi

  # 损失稳定判断
  cur=${fde:-0}
  prev=${prevfde[$((stage-1))]:-$cur}
  imp=$(python3 - "$prev" "$cur" <<'EOF'
import sys
try:
    p, c = float(sys.argv[1]), float(sys.argv[2])
except Exception:
    print(1.0); raise SystemExit
print(abs(p - c) / max(p, 1e-9))
EOF
)
  say "相对上轮 minFDE 变化=$imp"
  prevfde[$stage]=$cur
  if python3 -c "import sys; sys.exit(0 if float('$imp') < 0.005 else 1)"; then
    say "→ 损失已稳定（改善<0.5%），停止加训"
    break
  fi

  # 启动下一阶段
  lr=$(python3 -c "print(max(6e-5, 1.5e-4 * (0.7 ** ($stage-2))))")
  log "启动 stage$stage：+10 epoch，lr=$lr，从 $ck 续训"
  cd $R/tools || break
  CUDA_VISIBLE_DEVICES=1 VEC_INIT_CKPT="$ck" \
    $PY train_plant2_occ.py --occ-aux --lam-occ 0.05 --aux-pos-weight 20 \
    --data $D/vector_ds_occ --out $D/ckpt_occposaux$stage \
    --epochs 10 --bs 48 --lr "$lr" --lam-ce 0.5 --patience 3 \
    >> $D/train_stage$stage.log 2>&1
  stage=$((stage+1))
done

final=$(pick_ckpt)
if [ -n "$final" ]; then
  mkdir -p $D/ckpt_occposaux_final
  cp -f "$final" $D/ckpt_occposaux_final/best.pt
  say "最终模型已保存：$D/ckpt_occposaux_final/best.pt（来自 $final）"
fi
say "==== 流水线结束 $(date)  最后 Δ=$last_delta ===="
