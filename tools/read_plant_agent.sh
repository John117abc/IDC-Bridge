#!/bin/bash
F=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/plant_agent.py
echo "===== 1-60 ====="
sed -n '1,60p' "$F"
echo "===== 100-125 ====="
sed -n '100,125p' "$F"
echo "===== 190-260 ====="
sed -n '190,260p' "$F"
