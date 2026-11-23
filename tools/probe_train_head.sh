#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '240,305p' /gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py"
