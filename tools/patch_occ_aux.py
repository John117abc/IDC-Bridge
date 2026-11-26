"""P3：给规划器加占据栅格监督（aux 重建 16×16）

改动（全部向后兼容，默认关闭）：
  Planner.__init__ 增加 occ_aux 参数 → self.occ_aux 头（d → 256 → 16*16）
  forward 里用 occ_tok 的均值预测 16×16 占据，存 self.last_aux
  train_plant2_occ.py：--occ-aux 开关 + λ 权重，目标 = max_pool4(occ) 的任意通道
"""
import io

TP = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/train_planner.py'
TR = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/train_plant2_occ.py'

# ---------- 1) Planner ----------
s = io.open(TP, encoding='utf-8', errors='replace').read()

if 'occ_aux' not in s:
    s = s.replace(
        "    def __init__(self, K=K_DEF, d=256, layers=4, heads=8, dropout=0.1,\n"
        "                 use_occ=False, occ_pos=False):",
        "    def __init__(self, K=K_DEF, d=256, layers=4, heads=8, dropout=0.1,\n"
        "                 use_occ=False, occ_pos=False, occ_aux=False):", 1)

    s = s.replace(
        "        if self.occ_pos:\n",
        "        self.occ_aux = bool(occ_aux) and self.use_occ\n"
        "        if self.occ_aux:\n"
        "            # aux：从 occ token 重建 16x16 占据（强制编码器保留空间信息）\n"
        "            self.occ_aux_head = nn.Sequential(\n"
        "                nn.Linear(d, 256), nn.ReLU(inplace=True),\n"
        "                nn.Linear(256, 16 * 16))\n"
        "        if self.occ_pos:\n", 1)

    # forward：在 cat 进 tok 之前算 aux
    s = s.replace(
        "            tok = torch.cat([tok, occ_tok], 1)",
        "            if getattr(self, 'occ_aux', False):\n"
        "                self.last_aux = self.occ_aux_head(\n"
        "                    occ_tok.mean(1)).view(B, 16, 16)\n"
        "            tok = torch.cat([tok, occ_tok], 1)", 1)
    io.open(TP, 'w', encoding='utf-8', newline='').write(s)
    print('train_planner.py: occ_aux 已加')
else:
    print('train_planner.py: 已有 occ_aux')

# ---------- 2) 训练脚本 ----------
t = io.open(TR, encoding='utf-8', errors='replace').read()
if 'occ_aux' not in t:
    import re
    # 加 argparse 参数（找 args 定义区）
    m = re.search(r"(argparse\.ArgumentParser\([^\n]*\)\n)", t)
    if m:
        add_args = ('\nap.add_argument("--occ-aux", action="store_true",\n'
                    '                help="占据栅格 aux 重建监督")\n'
                    'ap.add_argument("--lam-occ", type=float, default=0.1)\n')
        t = t[:m.end()] + add_args + t[m.end():]
    else:
        print('警告：未找到 ArgumentParser，需手工加参数')

    # 模型构造处传 occ_aux
    t = re.sub(r"Planner\(([^)]*?)\)",
               lambda mm: ("Planner(" + mm.group(1) +
                           (", occ_aux=True" if 'occ_aux' not in mm.group(1) else "") + ")"),
               t, count=1)

    # loss 计算：在 backward 前加 aux
    old = "                loss = loss_wta + args.lam_ce * loss_ce"
    new = ("                loss = loss_wta + args.lam_ce * loss_ce\n"
           "                if getattr(args, 'occ_aux', False) and hasattr(model, 'last_aux'):\n"
           "                    _tgt = (F.max_pool2d(occ.float(), 4).sum(1) > 0).float()\n"
           "                    _aux = model.last_aux\n"
           "                    if _aux.shape[0] == _tgt.shape[0]:\n"
           "                        _laux = F.binary_cross_entropy_with_logits(_aux, _tgt)\n"
           "                        loss = loss + args.lam_occ * _laux\n"
           "                        last_aux_loss = float(_laux)\n")
    if old in t:
        t = t.replace(old, new, 1)
    else:
        print('警告：未匹配 loss 行，需手工插入')

    io.open(TR, 'w', encoding='utf-8', newline='').write(t)
    print('train_plant2_occ.py: --occ-aux 已加')
else:
    print('train_plant2_occ.py: 已有 occ_aux')
