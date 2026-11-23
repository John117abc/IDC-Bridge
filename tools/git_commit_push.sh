#!/bin/bash
set -u

echo "================ IDC-Bridge ================"
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
git add tools src
git status --porcelain | head -5
git -c user.name="jiangchengxuan" -c user.email="jiangchengxuan@users.noreply.github.com" \
    commit -q -m "feat(closedloop): Bench2Drive 两段式联调 —— IDC 84维 + PID 执行层 + CARLA BC 微调

- tools/frames.py: canonical 坐标系与 84 维状态/82 维编码（与训练端 bit-exact）
- tools/idc_carla.py: 按文件路径加载 actor，列拷贝热启动，RunningNorm 对齐
- tools/plant_agent_idc.py: CARLA 侧 IDC 控制器（route/规划器参考、曲率限速、安全层）
- tools/carla_pid_exec.py: PlanT 口径的横向/纵向 PID 执行层
- tools/bc_finetune.py: 用 CARLA 纯追踪数据做行为克隆微调
- src: 84 维状态接口（idc_agent / idc_state_builder / planner_runtime / checkpoint）
- 4 城镇闭环：2/4 满分（与 CARLA 官方 TM 持平），横向误差 0.12 m" || echo "(nothing to commit or failed)"
git log --oneline -1
git push -q origin HEAD 2>&1 | tail -3 && echo "IDC-Bridge pushed" || echo "IDC-Bridge push FAILED"

echo "================ plant2 ================"
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
cp -f /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/carla_pid_exec.py carla_garage/carla_pid_exec.py
git add carla_garage/frames.py carla_garage/idc_carla.py carla_garage/idc_state_carla.py \
        carla_garage/plant_agent_idc.py carla_garage/plant_agent_vec.py \
        carla_garage/carla_pid_exec.py .gitignore
git status --porcelain | head -8
git -c user.name="jiangchengxuan" -c user.email="jiangchengxuan@users.noreply.github.com" \
    commit -q -m "feat(carla): 接入 IDC 84维控制器与 PID 执行层（两段式闭环）

- plant_agent_vec.py: VEC_CTRL=idc/pp/tm 三种执行器；卡死提前终止；BC 数据录制
- plant_agent_idc.py: route/规划器参考、曲率限速、走廊约束、安全层
- carla_pid_exec.py: PlanT 参数横向/纵向 PID
- frames.py / idc_carla.py / idc_state_carla.py: 与训练端一致的状态构建" || echo "(nothing to commit or failed)"
git log --oneline -1
git push -q origin HEAD 2>&1 | tail -3 && echo "plant2 pushed" || echo "plant2 push FAILED"
