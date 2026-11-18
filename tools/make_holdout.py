"""生成"训练-留出"评测清单：复现建库时的 60k 采样，取补集作为 holdout。"""
import glob
import json
import os

import numpy as np

DATA = '/data/data_json/training'
N_TRAIN = 60000
SEED = 0
OUT = '/workspace/data/holdout_files.json'

files = sorted(glob.glob(os.path.join(DATA, 'tfrecord-*.json')))
print('全部场景', len(files))
rng = np.random.default_rng(SEED)
idx = rng.choice(len(files), N_TRAIN, replace=False)
train_set = {files[i] for i in idx}
holdout = [os.path.basename(f) for f in files if f not in train_set]
print('训练用', len(train_set), ' 留出', len(holdout))
with open(OUT, 'w') as fh:
    json.dump(holdout, fh)
print('saved', OUT)
