"""运行时占据栅格：从 CARLA 世界真值画 3×64×64（与训练数据同口径）

栅格定义（必须与 make_occupancy.py 完全一致）：
  64×64，0.5 m/格
  x（前后）∈ [-8, +24] m，行索引 0→63
  y（左右）∈ [-16, +16] m，列索引 0→63
通道：0=车辆 1=行人/骑行者 2=静态障碍
"""
import math
import os

import numpy as np

RES = 0.5
NX, NY = 64, 64
X0, Y0 = -8.0, -16.0

VEH = ('vehicle.',)
WALK = ('walker.',)
STATIC = ('static.prop.',)


def _classify(type_id, cls_name=''):
    tid = (type_id or '').lower()
    cls = (cls_name or '').lower()
    if 'walker' in tid or 'pedestrian' in tid or cls in ('walker', 'pedestrian'):
        return 1
    # 静态网格（路边停放车辆）在训练里归车辆通道
    if tid.startswith('static.prop.mesh'):
        return 0
    if tid.startswith(STATIC):
        return 2
    if tid.startswith(VEH) or cls in ('car', 'truck', 'bus', 'van'):
        return 0
    return -1


def _fill(occ, ch, cx, cy, yaw, hx, hy):
    """把 ego 系下的矩形写入栅格"""
    c, s = math.cos(-yaw), math.sin(-yaw)
    xs = X0 + (np.arange(NX) + 0.5) * RES
    ys = Y0 + (np.arange(NY) + 0.5) * RES
    dx = xs[:, None] - cx
    dy = ys[None, :] - cy
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    occ[ch] |= ((np.abs(lx) <= hx) & (np.abs(ly) <= hy)).astype(np.uint8)


def rasterize(world, ego, max_dist=40.0):
    """返回 (3,64,64) uint8；ego 为自车 actor"""
    occ = np.zeros((3, NX, NY), np.uint8)
    et = ego.get_transform()
    ex, ey = et.location.x, et.location.y
    eh = math.radians(et.rotation.yaw)
    ce, se = math.cos(-eh), math.sin(-eh)
    for a in world.get_actors():
        if a.id == ego.id:
            continue
        tid = a.type_id or ''
        ch = _classify(tid)
        if ch < 0:
            continue
        try:
            at = a.get_transform()
            bb = a.bounding_box
        except Exception:
            continue
        dx, dy = at.location.x - ex, at.location.y - ey
        if math.hypot(dx, dy) > max_dist:
            continue
        # 世界 → ego 系
        cx = dx * ce - dy * se
        cy = dx * se + dy * ce
        ax, ay = math.cos(-math.radians(at.rotation.yaw)), math.sin(-math.radians(at.rotation.yaw))
        cyaw = math.atan2(ay, ax) - math.atan2(se, ce)
        cyaw = math.atan2(math.sin(cyaw), math.cos(cyaw))
        hx = max(float(bb.extent.x), 0.25)
        hy = max(float(bb.extent.y), 0.25)
        if ch == 1:          # 行人包围盒较小，略放大以可见
            hx, hy = hx * 1.6, hy * 1.6
        _fill(occ, ch, cx, cy, cyaw, hx, hy)
    return occ


def summarize(occ):
    """给调试用的摘要：三通道占比 + 前方 12 m 走廊内是否有静态障碍"""
    veh = float(occ[0].mean())
    ped = float(occ[1].mean())
    sta = float(occ[2].mean())
    # 前方 12 m（行 16~40）、横向 ±2 m（列 28~36）
    corr = occ[2, 16:40, 28:36]
    blocked = int(corr.sum() > 0)
    return dict(veh=round(veh, 4), ped=round(ped, 4), sta=round(sta, 4),
                blocked=blocked)
