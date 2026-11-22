#!/bin/bash
D=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_Dataset
echo "===== 顶层 ====="
ls -la "$D" 2>/dev/null | head -15
echo "===== 前 40 个条目 ====="
find "$D" -maxdepth 3 2>/dev/null | head -40
echo "===== 最深层的 route 目录长什么样 ====="
R=$(find "$D" -maxdepth 4 -type d -name '*Rep*' 2>/dev/null | head -1)
echo "route dir = $R"
[ -n "$R" ] && ls -la "$R" | head -20
[ -n "$R" ] && for sub in "$R"/*/; do echo "  [dir] $sub : $(ls "$sub" 2>/dev/null | head -4 | tr '\n' ' ')"; done
echo "===== 找 boxes / measurements / target_point ====="
find "$D" -maxdepth 6 -type d \( -name 'boxes' -o -name 'measurements' -o -name 'bev_semantics' \) 2>/dev/null | head -6
find "$D" -maxdepth 7 -type f \( -name '*.json' -o -name '*.json.gz' -o -name '*.npz' -o -name '*.pkl' -o -name '*.png' \) 2>/dev/null | head -12
echo "===== 总文件数/体积分布 ====="
find "$D" -maxdepth 4 -type d 2>/dev/null | head -10
du -sh "$D"/* 2>/dev/null | head -10
