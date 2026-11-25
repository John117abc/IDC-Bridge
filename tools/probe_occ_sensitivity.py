"""占栅敏感性探针：固定其余输入，只在 8×8 块级别交换施工物位置，比较 6 模态轨迹与置信度。

块约定：occ 编码器为 Conv2d(k=8, s=8) → 64 个 token 对应 64 个 8×8 块。
  块 (br, bc) 覆盖 行 8br..8br+7（x 方向）、列 8bc..8bc+7（y 方向）
  x = -8 + (row+0.5)*0.5 米；y = -16 + (col+0.5)*0.5 米
"""
import sys

import numpy as np
import torch

R = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge'
sys.path.insert(0, R + '/src')
sys.path.insert(0, R + '/tools')
from train_planner import Planner                      # noqa: E402
from train_plant2_small import add_route_tokens        # noqa: E402

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default='/mnt/2T_HDD/plant2_vector/ckpt_occ/best.pt')
_ap.add_argument('--data',
                 default='/mnt/2T_HDD/plant2_vector/vector_ds_occ/ConstructionObstacle.npz')
_ap.add_argument('--out', default='')
_args = _ap.parse_args()
CKPT = _args.ckpt
DATA = _args.data

d = np.load(DATA)
occ_all = d['occ']                     # (N,3,64,64)
st = occ_all[:, 2]
# 走廊：x∈[0,16) m → 行 16..47；y∈[-3,3] → 列 26..37
corr = st[:, 16:48, 26:38].reshape(len(st), -1).sum(1)
i = int(np.argmax(corr))
print(f'样本 {i}，走廊内静态格 {int(corr[i])}，该帧总静态格 {int(st[i].sum())}')

occ = occ_all[i].copy()                # (3,64,64)


def blocks_of(occ, ch=2):
    """返回每个 8×8 块的静态格数 (8,8)"""
    return occ[ch].reshape(8, 8, 8, 8).sum(axis=(1, 3))


b = blocks_of(occ)
# 目标块：走廊内（块行 2..5 → x 0~16 m；块列 3..4 → y -4~+4）
cand = [(r, c, int(b[r, c])) for r in range(2, 6) for c in range(3, 5)]
cand.sort(key=lambda t: -t[2])
src = cand[0]
print('走廊内块静态格数:', cand)
print(f'源块（有施工物）(br,bc)=({src[0]},{src[1]}) 格数={src[2]}')

# 目标空块：同一块行，列 0（y≈-16~-12，路边），要求为空
dst = None
for c in (0, 7):
    if b[src[0], c] == 0:
        dst = (src[0], c)
        break
if dst is None:
    raise SystemExit('找不到空块')
print(f'目标空块 (br,bc)={dst} 格数={int(b[dst[0], dst[1]])}')


def blk_meters(br, bc):
    x0 = -8 + br * 8 * 0.5
    x1 = x0 + 4.0
    y0 = -16 + bc * 8 * 0.5
    y1 = y0 + 4.0
    return (x0, x1, y0, y1)


print('  源块范围 x[%.1f,%.1f] y[%.1f,%.1f] m' % blk_meters(*src[:2]))
print('  目标块范围 x[%.1f,%.1f] y[%.1f,%.1f] m' % blk_meters(*dst))

# 整块交换（三通道一起换，保证障碍大小/数量完全一致）
A = occ.copy()
B = occ.copy()
for ch in range(3):
    sa = A[ch, src[0] * 8:src[0] * 8 + 8, src[1] * 8:src[1] * 8 + 8].copy()
    sb = B[ch, dst[0] * 8:dst[0] * 8 + 8, dst[1] * 8:dst[1] * 8 + 8].copy()
    A[ch, src[0] * 8:src[0] * 8 + 8, src[1] * 8:src[1] * 8 + 8] = sb     # 走廊块清空
    A[ch, dst[0] * 8:dst[0] * 8 + 8, dst[1] * 8:dst[1] * 8 + 8] = sa     # 障碍移到路边
    B[ch, src[0] * 8:src[0] * 8 + 8, src[1] * 8:src[1] * 8 + 8] = sa     # 对照：障碍留在走廊

print('交换后通道2:')
print('  A（障碍在路边）走廊块格数', int(blocks_of(A)[src[0], src[1]]),
      '路边块', int(blocks_of(A)[dst[0], dst[1]]))
print('  B（障碍在走廊）走廊块格数', int(blocks_of(B)[src[0], src[1]]),
      '路边块', int(blocks_of(B)[dst[0], dst[1]]))

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
print('ckpt use_occ =', ck.get('use_occ'))
model = Planner(K=int(ck.get('K', 6)), use_occ=True).to(dev)
model.load_state_dict(ck['model'], strict=False)
model.emit_logits = True
model.eval()

ego = torch.tensor(d['ego'][i:i + 1], device=dev)
ag = torch.tensor(d['agent'][i:i + 1], device=dev)
am = torch.tensor(d['amask'][i:i + 1], device=dev)
mp0 = torch.tensor(d['map'][i:i + 1], device=dev)
mm0 = torch.tensor(d['mmask'][i:i + 1], device=dev)
rt = torch.tensor(d['route'][i:i + 1], device=dev)
MP, MM = add_route_tokens(mp0.cpu().numpy(), mm0.cpu().numpy(), rt.cpu().numpy())
MP = torch.tensor(MP, device=dev)
MM = torch.tensor(MM, device=dev)

with torch.no_grad():
    oA = torch.tensor(A[None]).float().to(dev)
    oB = torch.tensor(B[None]).float().to(dev)
    pA, lA = model(ego, ag, am, MP, MM, occ=oA)
    pB, lB = model(ego, ag, am, MP, MM, occ=oB)
    pA, lA, pB, lB = pA.cpu().numpy()[0], lA.cpu().numpy()[0], pB.cpu().numpy()[0], lB.cpu().numpy()[0]

print('\n=== 六模态轨迹差（A 障碍在路边 vs B 障碍在走廊）===')
diff = np.abs(pA - pB)
print('  全局 max|Δ| = %.5f m   mean|Δ| = %.5f m' % (diff.max(), diff.mean()))
for k in range(6):
    print('  mode%d  终点(%.3f,%.3f) vs (%.3f,%.3f)  |Δ|max=%.5f   置信度 %.4f vs %.4f'
          % (k, pA[k, -1, 0], pA[k, -1, 1], pB[k, -1, 0], pB[k, -1, 1],
             diff[k].max(), lA[k], lB[k]))

print('\n=== 选择与接口量 ===')
kA, kB = int(np.argmax(lA)), int(np.argmax(lB))
print('  argmax 模态: A=%d  B=%d' % (kA, kB))
print('  置信度最大差 = %.5f' % np.abs(lA - lB).max())
# 当前接口：8 点横向偏移中位数
medA = float(np.median(pA[kA, :, 1]))
medB = float(np.median(pB[kB, :, 1]))
print('  接口量（中位横向偏移）：A=%.4f m  B=%.4f m  差=%.4f m' % (medA, medB, medB - medA))

if _args.out:
    with open(_args.out, 'w') as _f:
        _f.write('max_dpred=%.6f\n' % diff.max())
        _f.write('max_dlogit=%.6f\n' % np.abs(lA - lB).max())
        _f.write('argmax_A=%d argmax_B=%d\n' % (kA, kB))
        _f.write('ckpt=%s\n' % CKPT)
    print('已写结论', _args.out)
