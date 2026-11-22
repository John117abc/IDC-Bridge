#!/bin/bash
set -u
C=/mnt/2T_HDD/CARLA_0.9.16
PY=/mnt/2T_HDD/conda_envs/bench2drive_b2d/bin/python
L=/mnt/2T_HDD/plant2_vector

echo "=== 已有 CARLA 进程 ==="
pgrep -af CarlaUE4 | head -3
echo "=== 可执行文件 ==="
ls -la "$C/CarlaUE4.sh" 2>/dev/null

echo "=== 启动 CARLA（headless, port 2000）==="
nohup "$C/CarlaUE4.sh" -RenderOffScreen -nosound -carla-port=2000 \
  -quality-level=Low -stdout > "$L/carla_server.log" 2>&1 &
sleep 50

echo "=== 端口监听 ==="
(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep 2000 || echo "2000 未监听"
echo "=== 进程 ==="
pgrep -af CarlaUE4 | head -3
echo "=== 日志尾部 ==="
tail -12 "$L/carla_server.log" 2>/dev/null

echo "=== 客户端连接测试 ==="
timeout 90 "$PY" -c "
import carla, time
for i in range(10):
    try:
        c = carla.Client('127.0.0.1', 2000); c.set_timeout(10.0)
        w = c.get_world()
        print('OK map =', w.get_map().name, '| actors =', len(w.get_actors()))
        print('carla ver =', c.get_server_version())
        break
    except Exception as e:
        print('retry', i, type(e).__name__, e); time.sleep(5)
" 2>&1 | tail -6
