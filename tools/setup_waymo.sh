#!/usr/bin/env bash
P=/gpudrive/.venv/bin/python
echo "--- TF 版本 ---"
$P -c "import tensorflow as tf; print(tf.__version__)" 2>/dev/null | tail -1
echo "--- ensurepip ---"
$P -m ensurepip --upgrade 2>&1 | tail -2
echo "--- pip ---"
$P -m pip --version 2>&1 | tail -1
echo "--- 已装 waymo? ---"
$P -c "import waymo_open_dataset, os; print(os.path.dirname(waymo_open_dataset.__file__))" 2>&1 | tail -1
