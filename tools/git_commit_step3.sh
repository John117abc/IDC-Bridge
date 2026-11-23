#!/bin/bash
set -u
echo "================ IDC-Bridge ================"
cd /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge
cp -f /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/*.py /tmp/ 2>/dev/null || true
git add tools src
git -c user.name="jiangchengxuan" -c user.email="jiangchengxuan@users.noreply.github.com" \
    commit -q -m "feat(step3): 动作空间改为 (横向偏移, 加速度意图)

- tools/bc_2d.py: 从 CARLA BC 数据蒸馏 2D 动作空间（off/acc），30 ep，loss 0.146
- tools/idc_carla.py: 读取 checkpoint 的 action_space 标记
- tools/plant_agent_idc.py: 2D 分支——off 直接横移 route，acc 积分成速度指令
- tools/run_step3.sh: 4 城镇验证脚本
- 结果：完成数 2/4 → 3/4，横向误差 0.18 m" || echo "(nothing to commit)"
git log --oneline -1
git push -q origin HEAD 2>&1 | tail -2 && echo "IDC-Bridge pushed" || echo "push FAILED"

echo "================ plant2 ================"
cd /home/jiangchengxuan/PycharmProjects/research-safe-drive/plant2
for f in idc_carla plant_agent_idc frames idc_state_carla carla_pid_exec plant_agent_vec; do
  [ -f /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/$f.py ] && \
    cp -f /home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/$f.py carla_garage/$f.py
done
git add carla_garage/idc_carla.py carla_garage/plant_agent_idc.py carla_garage/plant_agent_vec.py \
        carla_garage/frames.py carla_garage/idc_state_carla.py carla_garage/carla_pid_exec.py
git -c user.name="jiangchengxuan" -c user.email="jiangchengxuan@users.noreply.github.com" \
    commit -q -m "feat(step3): 同步 2D 动作空间（off/acc）到 carla_garage" || echo "(nothing to commit)"
git log --oneline -1
git push -q origin HEAD 2>&1 | tail -2 && echo "plant2 pushed" || echo "push FAILED"
