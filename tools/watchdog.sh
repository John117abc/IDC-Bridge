#!/bin/bash
# 看门狗：容器/GPU 掉了就重起容器，训练进程全没了就重启对应阶段（从 checkpoint 续）
set -u
L=/workspace/data/logs
LOG=$L/watchdog.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }

say "watchdog start"
for i in $(seq 1 400); do
  # 1) 容器
  st=$(docker inspect -f '{{.State.Running}}' idc_hist 2>/dev/null || echo false)
  if [ "$st" != "true" ]; then
    say "container down -> start"
    docker start idc_hist >> $LOG 2>&1
    sleep 20
  fi
  # 2) GPU
  if ! docker exec idc_hist nvidia-smi -L > /dev/null 2>&1; then
    say "GPU unavailable -> restart container"
    docker restart idc_hist >> $LOG 2>&1
    sleep 25
  fi
  # 3) 训练进程
  np1=$(pgrep -f run_p1.sh | wc -l)
  np2=$(pgrep -f run_p2.sh | wc -l)
  ntr=$(docker exec idc_hist pgrep -f train_planner | wc -l)
  done1=$([ -f /workspace/data/p1_iawr/latest.pt ] && echo 1 || echo 0)
  done2=$([ -f /workspace/data/p2_anchor_iawr/latest.pt ] && echo 1 || echo 0)
  if [ "$np1" -eq 0 ] && [ "$np2" -eq 0 ] && [ "$ntr" -eq 0 ]; then
    if [ "$done1" -eq 0 ]; then
      say "P1 中途断了 -> 重启 run_p1.sh"
      nohup bash /gpudrive/baselines/IDC-Bridge/tools/run_p1.sh >> $L/p1_restart.log 2>&1 &
    elif [ "$done2" -eq 0 ]; then
      say "P2 中途断了 -> 重启 run_p2.sh"
      nohup bash /gpudrive/baselines/IDC-Bridge/tools/run_p2.sh >> $L/p2_restart.log 2>&1 &
    fi
  fi
  sleep 300
done
say "watchdog exit"
