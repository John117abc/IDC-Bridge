#!/bin/bash
# 暂停训练：停掉 stage2、接力、看门狗、流水线（保留 stage1 的 ckpt）
set -u
D=/mnt/2T_HDD/plant2_vector
SELF=$$
for pat in "train_plant2_occ.py" "train_watch" "p3_auto_pipeline" "stage2 结束"; do
  for pid in $(pgrep -f "$pat" 2>/dev/null); do
    [ "$pid" = "$SELF" ] && continue
    kill -9 "$pid" 2>/dev/null
  done
done
sleep 2
echo "剩余训练进程: $(pgrep -f train_plant2_occ | wc -l)"
echo "stage1 ckpt:"
ls -la --time-style=+%H:%M $D/ckpt_occposaux/best.pt 2>/dev/null
echo "stage1 最终指标:"
grep "训练完成" $D/train_occposaux.log | tail -1
