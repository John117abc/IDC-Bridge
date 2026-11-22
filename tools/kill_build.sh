#!/bin/bash
echo "=== 构建进程状态 ==="
ps -o pid,etime,time,pcpu,cmd -p $(pgrep -f build_plant2_dataset | head -1) 2>/dev/null
echo "=== 若已无进展则结束它 ==="
pkill -f run_full_all.sh 2>/dev/null
pkill -f build_plant2_dataset 2>/dev/null
sleep 2
echo "剩余: $(pgrep -f build_plant2_dataset | wc -l)"
echo "npz: $(ls /mnt/2T_HDD/plant2_vector/vector_ds_full | wc -l)"
