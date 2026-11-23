#!/bin/bash
D=/mnt/2T_HDD/plant2_vector
echo "=== 链式任务进度 ==="
cat $D/chain_hist.log 2>/dev/null
echo "=== 进程 ==="
echo "构建: $(pgrep -f build_plant2_dataset | wc -l)  训练(WTA): $(pgrep -f train_plant2_full | wc -l)  训练(conf): $(pgrep -f train_plant2_conf | wc -l)"
echo "=== 数据集 ==="
ls $D/vector_ds_hist 2>/dev/null | wc -l
du -sh $D/vector_ds_hist 2>/dev/null
echo "=== 训练日志（若已开始）==="
tail -5 $D/train_hist.log 2>/dev/null || echo "train_hist.log 还不存在（仍在构建数据集）"
echo "=== GPU ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
