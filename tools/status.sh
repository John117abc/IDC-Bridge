#!/bin/bash
D=/mnt/2T_HDD/plant2_vector/vector_ds_full
L=/mnt/2T_HDD/plant2_vector
echo "=== npz 文件数 ==="
ls "$D" | wc -l
du -sh "$D"
echo "=== 构建进程 ==="
pgrep -f build_plant2_dataset | wc -l
echo "=== finish_train.log ==="
tail -12 "$L/finish_train.log" 2>/dev/null
echo "=== train_full.log ==="
tail -12 "$L/train_full.log" 2>/dev/null
echo "=== 训练进程 ==="
pgrep -f train_plant2_small | wc -l
