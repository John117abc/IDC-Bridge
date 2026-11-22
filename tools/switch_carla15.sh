#!/bin/bash
set -u
C15=/home/jiangchengxuan/carla-0.9.15-linux
PY37=/mnt/2T_HDD/conda_envs/bench2drive/bin/python
L=/mnt/2T_HDD/plant2_vector

echo "=== 停掉 0.9.16 服务端 ==="
pkill -f CarlaUE4-Linux-Shipping 2>/dev/null
pkill -f CarlaUE4.sh 2>/dev/null
sleep 5
pgrep -af CarlaUE4 | head -3 || echo "已全部停止"

echo "=== 启动 CARLA 0.9.15（headless, port 2000）==="
nohup "$C15/CarlaUE4.sh" -RenderOffScreen -nosound -carla-port=2000 \
  -quality-level=Low -stdout > "$L/carla15.log" 2>&1 &
sleep 55
(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep 2000 || echo "2000 未监听"
pgrep -af CarlaUE4-Linux-Shipping | head -2

echo "=== py3.7(bench2drive) 客户端连接测试 ==="
timeout 120 "$PY37" - <<'EOF' 2>&1 | tail -8
import time
import carla
print('client carla version =', carla.__version__)
for i in range(8):
    try:
        c = carla.Client('127.0.0.1', 2000); c.set_timeout(10.0)
        w = c.get_world()
        print('OK map =', w.get_map().name)
        print('server =', c.get_server_version())
        print('actors =', len(w.get_actors()))
        break
    except Exception as e:
        print('retry', i, type(e).__name__, str(e)[:120]); time.sleep(6)
EOF
