#!/bin/bash
echo "--- container root ---"
docker exec idc_hist ls /
echo "--- /gpudrive ---"
docker exec idc_hist ls /gpudrive | head -8
echo "--- /workspace ---"
docker exec idc_hist ls /workspace 2>&1 | head -8
echo "--- mounts ---"
docker exec idc_hist sh -c 'cat /proc/mounts | grep -E "workspace|gpudrive"' | head -5
echo "--- 宿主机 IDC-Bridge 是否挂进容器 ---"
docker inspect idc_hist --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'
