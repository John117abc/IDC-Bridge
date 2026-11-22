#!/bin/bash
R=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
echo "===== carla_garage/data.py 结构 ====="
grep -nE "^class |^def |^    def " "$R/carla_garage/data.py" | head -40
echo "===== 数据文件里存了什么 ====="
grep -nE "self\.|np\.|keys\(\)|'ego'|\"ego\"|route|lane|bev|object" "$R/carla_garage/data.py" | head -40
echo "===== data_agent.py 关键函数 ====="
grep -nE "^class |^def |^    def " "$R/carla_garage/data_agent.py" | head -30
echo "===== 数据保存格式（npz/pkl 字段） ====="
grep -nE "npz|pickle|savez|\.pkl|dump\(" "$R/carla_garage/data_agent.py" "$R/carla_garage/data.py" | head -20
echo "===== PlanT/dataset.py 读什么 ====="
grep -nE "^class |^    def |keys|\[.*\]|route|lane|waypoint" "$R/PlanT/dataset.py" | head -30
echo "===== 现有数据集样本 ====="
grep -rIn --include=*.py -e "data_dir" -e "dataset_path" "$R/carla_garage/train.py" 2>/dev/null | head -8
find "$R" -maxdepth 3 -type d -name 'data' | head -5
du -sh "$R/data"/* 2>/dev/null | head -8
