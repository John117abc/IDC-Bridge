#!/usr/bin/env bash
P=/gpudrive/.venv/bin/python
echo "=== 试 2-18-0（显示完整错误）==="
$P -m pip install --no-deps waymo-open-dataset-tf-2-18-0 2>&1 | grep -vE "^\[notice\]" | tail -8
echo "=== 试直接装 waymo-open-dataset ==="
$P -m pip install --no-deps waymo-open-dataset 2>&1 | grep -vE "^\[notice\]" | tail -5
echo "=== pip 能否列出可用版本 ==="
$P -m pip index versions waymo-open-dataset-tf-2-16-0 2>&1 | grep -vE "^\[notice\]" | tail -5
