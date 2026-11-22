#!/bin/bash
set -u
PY37=/mnt/2T_HDD/conda_envs/bench2drive/bin/python
timeout 120 "$PY37" - <<'EOF' 2>&1 | tail -8
import time
import carla
print('carla module ok')
for i in range(8):
    try:
        c = carla.Client('127.0.0.1', 2000); c.set_timeout(10.0)
        w = c.get_world()
        print('OK map =', w.get_map().name)
        print('server =', c.get_server_version())
        print('actors =', len(w.get_actors()))
        break
    except Exception as e:
        print('retry', i, type(e).__name__, str(e)[:160]); time.sleep(6)
EOF
