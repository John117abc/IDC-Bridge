#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "===== 训练/数据入口 ====="
ls "$R"/gate5 "$R"/p1 2>/dev/null | head -30
echo "===== 找 train 脚本 ====="
find "$R" -maxdepth 3 -name 'train*.py' -o -maxdepth 3 -name '*dataset*.py' -o -maxdepth 3 -name '*data*.py' | grep -v third_party | head -20
echo "===== PlanT2 模型/agent 文件 ====="
find "$R" -maxdepth 4 -name '*.py' | grep -iE 'plant2|plant|agent' | grep -v third_party | head -20
echo "===== proxy_simulator ====="
ls -la "$R/tools/proxy_simulator" 2>/dev/null | head -15
echo "===== 我们的输入是否可得：grep ====="
for k in "get_actors" "get_map" "GlobalRoutePlanner" "route" "lane_center" "OpenDRIVE" "waypoint"; do
  n=$(grep -rIl --include=*.py "$k" "$R" 2>/dev/null | grep -v third_party | wc -l)
  echo "$k -> $n 个文件"
done
echo "===== agent 基类里拿周车/地图的地方 ====="
grep -rIn --include=*.py -e "world.get_actors" -e "get_map()" -e "get_waypoint" "$R" 2>/dev/null | grep -v third_party | head -12
