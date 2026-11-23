"""V1：检查 dump 出来的规划器参考是否合法（NaN / 后退 / 长度异常）。"""
import numpy as np

z = np.load('/workspace/data/plan_dump.npz')
print('keys:', z.files)
for k in z.files:
    a = z[k]
    print(' ', k, getattr(a, 'shape', None), a.dtype)

# dumps[w] = dict(steps=[], cands=[], sel=[], tgt=[], ego=[])
# _save_dump 的落盘格式未知，先把 ndarray 类字段统计一下
for k in z.files:
    a = np.asarray(z[k], dtype=object) if z[k].dtype == object else z[k]
    try:
        arr = np.asarray(a, dtype=np.float64)
    except Exception:
        continue
    if arr.ndim < 2:
        continue
    finite = np.isfinite(arr)
    print(f'\n[{k}] 形状 {arr.shape}  NaN/inf {int((~finite).sum())}')
    print('   min %.2f  max %.2f  mean %.2f' % (
        np.nanmin(arr), np.nanmax(arr), np.nanmean(arr)))
