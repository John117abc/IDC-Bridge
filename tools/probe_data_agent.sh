#!/bin/bash
F=/home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2/carla_garage/data_agent.py
echo "===== tick_data 返回的键 ====="
grep -nE "^\s+'[a-z_]+':|tick_data\s*=|return tick_data" "$F" | head -30
echo "===== 自车/世界属性 ====="
grep -nE "self\._vehicle|self\._world|self\.world\b|get_actors\(\)" "$F" | head -12
echo "===== run_step 签名 ====="
grep -nE "def run_step|def setup\(" "$F" | head -5
echo "===== route/target_point 来源 ====="
grep -nE "route|target_point|global_plan" "$F" | head -20
