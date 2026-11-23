#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '210,240p' /gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py"
