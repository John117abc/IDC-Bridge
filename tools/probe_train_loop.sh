#!/bin/bash
set -u
docker exec idc_hist bash -lc "sed -n '300,400p' /gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py"
