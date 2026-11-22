#!/bin/bash
Z=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_Dataset/PlanT2_DS.zip
D2=/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_DS
echo "===== zip 里前 40 条 ====="
unzip -l "$Z" 2>/dev/null | head -40
echo "===== zip 里各级目录名统计 ====="
unzip -l "$Z" 2>/dev/null | awk '{print $4}' | grep -oE '^[^/]+/[^/]*/' | sort -u | head -15
echo "===== zip 里关键的 boxes / measurements / target_point ====="
unzip -l "$Z" 2>/dev/null | awk '{print $4}' | grep -iE 'boxes|measurement|target_point|bev' | head -12
echo "===== 已解压的 PlanT2_DS ====="
ls -la "$D2" 2>/dev/null | head -12
find "$D2" -maxdepth 3 2>/dev/null | head -20
