#!/bin/bash
set -u
echo "===== IDC-Bridge ====="
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge || exit 1
git branch --show-current
git status --porcelain | grep -E "plant_agent_idc|carla_pid_exec|idc_state_carla|train_planner|train_plant2_occ|eval_ckpt|eval_panel_plan2|probe_occ|analyze_long|patch_long|patch_aux|patch_occ|p3_|save_model|morning_report|status_now|health_check" | head -30
echo
echo "===== plant2 ====="
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2 || exit 1
git branch --show-current 2>/dev/null || echo "(非 git 或子模块)"
git status --porcelain 2>/dev/null | grep -E "carla_garage/(plant_agent_idc|plant_agent_vec|carla_pid_exec|idc_state_carla)" | head -10
