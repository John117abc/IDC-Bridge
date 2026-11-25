"""从 PlanT2 数据集的 boxes/*.gz 生成 3 通道占据栅格。

栅格：64×64，0.5 m/格；ego 在中心
  前后：x ∈ [-8, +24] m（前向偏置，规划更需要前方）
  左右：y ∈ [-16, +16] m
通道：
  0 = 车辆（类 car/truck/bus/van/bike 等 vehicle.*）
  1 = 行人（pedestrian / cyclist）
  2 = 静态障碍（static.prop.*，施工锥桶/围挡/碎屑）

输出：每个 route 目录一个 .npz（frame_ids, occ）
"""
import gzip
import json
import os
import sys
import zipfile

import numpy as np

Z = '/mnt/2T_HDD/PlanT2.0/PlanT2_Dataset/data/PlanT2_Dataset/PlanT2_DS.zip'
OUT = '/mnt/2T_HDD/plant2_vector/occupancy'

RES = 0.5
NX, NY = 64, 64            # x: 前方向 64 格，y: 横向 64 格
X0, Y0 = -8.0, -16.0       # 起点


def poly_mask(pos, yaw, hx, hy):
    """返回栅格内的布尔掩码（矩形与栅格相交判断，逐格中心）"""
    cx, cy = float(pos[0]), float(pos[1])
    c, s = np.cos(-yaw), np.sin(-yaw)
    xs = X0 + (np.arange(NX) + 0.5) * RES
    ys = Y0 + (np.arange(NY) + 0.5) * RES
    XXX, YYY = np.meshgrid(xs, ys, indexing='ij')     # [NX, NY]
    dx = XXX - cx
    dy = YYY - cy
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return (np.abs(lx) <= hx) & (np.abs(ly) <= hy)


def classify(it):
    """返回通道：0=车辆 1=行人/骑行者 2=静态障碍 -1=忽略

    数据集的标注口径（实测）：
      car / ego_car            → 车辆
      static_car + static.prop.mesh → 路边停放车辆（按车辆处理）
      walker (type_id=None)    → 行人
      static.prop.constructioncone / trafficwarning / dirtdebris → 静态障碍
      traffic_light / stop_sign → 忽略
    """
    cls = (it.get('class') or '').lower()
    tid = (it.get('type_id') or '').lower()
    if cls in ('walker', 'pedestrian', 'cyclist', 'rider') or \
            'walker' in tid or 'pedestrian' in tid:
        return 1
    if cls == 'static_car':
        return 0
    if tid.startswith('static.prop') and 'mesh' not in tid:
        return 2
    if cls in ('car', 'truck', 'bus', 'van', 'motorcycle', 'bicycle') or \
            tid.startswith('vehicle.'):
        return 0
    return -1


def rasterize(boxes):
    occ = np.zeros((3, NX, NY), np.uint8)
    for it in boxes:
        ch = classify(it)
        if ch < 0:
            continue
        if (it.get('class') or '').lower() == 'ego_car':
            continue
        ext = it.get('extent') or [1.0, 0.5, 0.5]
        m = poly_mask(it.get('position', [0, 0, 0]), float(it.get('yaw', 0.0)),
                      float(ext[0]), float(ext[1]))
        occ[ch] |= m.astype(np.uint8)
    return occ


def main():
    os.makedirs(OUT, exist_ok=True)
    with zipfile.ZipFile(Z) as z:
        names = z.namelist()
        routes = {}
        for n in names:
            if '/boxes/' in n and n.endswith('.json.gz'):
                route = n.split('/boxes/')[0]
                routes.setdefault(route, []).append(n)
        print(f'共 {len(routes)} 个 route 目录', flush=True)

        done = 0
        for route, files in sorted(routes.items()):
            rel = route.replace('PlanT_2_dataset/data/', '')
            dst = os.path.join(OUT, rel.replace('/', '__') + '.npz')
            if os.path.exists(dst) and os.environ.get('FORCE', '0') != '1':
                done += 1
                continue
            files = sorted(files)
            fids, occs = [], []
            for f in files:
                try:
                    obj = json.loads(gzip.decompress(z.read(f)).decode())
                except Exception:
                    continue
                fids.append(int(os.path.basename(f).split('.')[0]))
                occs.append(rasterize(obj))
            if not fids:
                continue
            np.savez_compressed(dst, frame_ids=np.array(fids, np.int32),
                                occ=np.stack(occs))
            done += 1
            if done % 20 == 0:
                print(f'  已处理 {done}/{len(routes)}: {rel}', flush=True)
        print(f'完成，共 {done} 个 route', flush=True)


if __name__ == '__main__':
    main()
