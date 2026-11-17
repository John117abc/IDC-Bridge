import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else '/workspace/data/head_dd/anchors.npy'
A = np.load(path)
print('file', path, 'shape', A.shape)
L = np.linalg.norm(A[:, -1, :], axis=1)
Y = np.abs(A[:, :, 1]).max(1)
print('终点距离(m):', np.round(np.sort(L), 1).tolist())
print('最大横向(m):', np.round(np.sort(Y), 2).tolist())
