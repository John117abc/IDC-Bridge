#!/bin/bash
B=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS/PlanT_2_dataset/data
R=$(find "$B" -maxdepth 2 -mindepth 2 -type d | head -1)
echo "route dir = $R"
ls -la "$R" | head -20
echo "===== 子目录清单 ====="
for s in "$R"/*/; do echo "  $(basename $s) : $(ls "$s" | wc -l) 个文件, 例: $(ls "$s" | head -2 | tr '\n' ' ')"; done
echo "===== 一个 boxes json 的内容 ====="
F=$(ls "$R"/boxes/*.json.gz 2>/dev/null | head -1)
echo "file=$F"
[ -n "$F" ] && zcat "$F" | head -c 900
echo
echo "===== measurements 内容 ====="
M=$(ls "$R"/measurements/* 2>/dev/null | head -1)
echo "file=$M"
[ -n "$M" ] && (zcat "$M" 2>/dev/null | head -c 400 || head -c 400 "$M")
