#!/bin/bash
# 链式：等位置编码微调完成 → 跑探针 → 若位置敏感则 4 路线闭环（面板 gif）→ 通过则 24 路线
set -u
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
T=/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools
D=/mnt/2T_HDD/plant2_vector
LOG=$D/chain_probe_eval.log
REP=$D/chain_probe_eval_report.txt
CKPT=$D/ckpt_occpos/best.pt
PROBE_OUT=$D/probe_result.txt

echo "[$(date +%H:%M:%S)] 链式启动：等待位置编码微调完成" >> "$LOG"

# 1) 等微调
while true; do
  if grep -q "训练完成" "$D/train_occpos.log" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] 微调完成" >> "$LOG"; break
  fi
  if ! pgrep -f "train_plant2_occ.py .*ckpt_occpos" > /dev/null 2>&1 \
     && grep -q "ep " "$D/train_occpos.log" 2>/dev/null; then
    sleep 60
    if ! pgrep -f "train_plant2_occ.py .*ckpt_occpos" > /dev/null 2>&1; then
      echo "[$(date +%H:%M:%S)] 微调进程消失，视为结束" >> "$LOG"; break
    fi
  fi
  sleep 60
done

ls -la "$CKPT" >> "$LOG" 2>&1
grep -E "ep |早停|训练完成" "$D/train_occpos.log" | tail -4 >> "$LOG"

# 2) 探针
echo "[$(date +%H:%M:%S)] 跑整块交换探针" >> "$LOG"
$PY $T/probe_occ_sensitivity.py --ckpt "$CKPT" --out "$PROBE_OUT" >> "$LOG" 2>&1
tail -8 "$PROBE_OUT" >> "$LOG" 2>/dev/null
DP=$(grep -oP 'max_dpred=\K[0-9.]+' "$PROBE_OUT" 2>/dev/null || echo 0)
echo "[$(date +%H:%M:%S)] 探针 Δpred=$DP" >> "$LOG"
{
  echo "======== 位置编码微调后探针（$(date +%H:%M)）========"
  cat "$PROBE_OUT" 2>/dev/null
} >> "$REP" 2>&1

# 3) 位置敏感才闭环
SENS=$(awk -v d="$DP" 'BEGIN{print (d>0.01)?"1":"0"}')
if [ "$SENS" != "1" ]; then
  echo "[$(date +%H:%M:%S)] Δpred=$DP 未达阈值 0.01，跳过闭环（说明位置编码未被用起来）" >> "$LOG"
  echo "结论：位置编码未被使用，需转辅助碰撞损失/可行性筛选" >> "$REP"
  exit 0
fi

echo "[$(date +%H:%M:%S)] 位置敏感，进入 4 路线闭环" >> "$LOG"
bash $T/eval_with_panel.sh "$CKPT" occpos4 "$D/other_towns.xml" 600 >> "$LOG" 2>&1
RD=$(ls -dt $D/closedloop/occpos4_* 2>/dev/null | head -1)
{
  echo
  echo "======== 位置编码模型 4 路线（$(date +%H:%M)）========"
  bash $T/show_results.sh "$RD" 2>&1 | grep -E 'route Route|infractions' | head -8
} >> "$REP" 2>&1

N_OK=$(grep -c "Completed" "$REP" 2>/dev/null || echo 0)
echo "[$(date +%H:%M:%S)] 4 路线报告内 Completed 行数=$N_OK" >> "$LOG"
if [ "$N_OK" -ge 3 ]; then
  echo "[$(date +%H:%M:%S)] 进入 24 路线" >> "$LOG"
  bash $T/eval_with_panel.sh "$CKPT" occpos24 "$D/night_routes.xml" 420 >> "$LOG" 2>&1
  RD2=$(ls -dt $D/closedloop/occpos24_* 2>/dev/null | head -1)
  {
    echo
    echo "======== 位置编码模型 24 路线（$(date +%H:%M)）========"
    bash $T/show_results.sh "$RD2" 2>&1 | grep -E 'route Route' | head -26
  } >> "$REP" 2>&1
fi
echo "[$(date +%H:%M:%S)] 链式全部完成" >> "$LOG"
