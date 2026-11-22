#!/bin/bash
echo "=== checkpoints ==="
find /workspace/data/checkpoints -maxdepth 2 -type f | head -10
echo "=== world_files / 文件清单 json ==="
find /workspace/data -maxdepth 2 -type f -name '*.json' | head -10
echo "=== 其他 pth ==="
find /workspace/data -maxdepth 3 -type f -name '*.pth' | head -10
