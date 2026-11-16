import collections
import glob

import numpy as np

c = collections.Counter()
for f in glob.glob('/workspace/data/history/*.npz'):
    z = np.load(f, allow_pickle=True)
    c.update(z['current_time_index'].tolist())
    n = len(z['scenario_id'])
print('scenarios:', sum(c.values()), ' current_time_index 分布:', dict(c))
