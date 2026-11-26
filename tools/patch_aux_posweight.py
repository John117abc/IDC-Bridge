"""A：aux 修复——BCE 加 pos_weight（处理稀疏），并强制 occ_pos 训练"""
import io

TR = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/train_plant2_occ.py'
s = io.open(TR, encoding='utf-8', errors='replace').read()

# 1) 新增参数
if '--aux-pos-weight' not in s:
    s = s.replace(
        '    ap.add_argument("--lam-occ", type=float, default=0.1)',
        '    ap.add_argument("--lam-occ", type=float, default=0.1)\n'
        '    ap.add_argument("--aux-pos-weight", type=float, default=20.0,\n'
        '                    help="占据为正类的 BCE pos_weight（稀疏补偿）")', 1)

# 2) BCE 加 pos_weight
old = "                        _laux = F.binary_cross_entropy_with_logits(_aux, _tgt)"
new = ("                        _pw = torch.tensor(\n"
       "                            float(getattr(args, 'aux_pos_weight', 20.0)),\n"
       "                            device=_aux.device)\n"
       "                        _laux = F.binary_cross_entropy_with_logits(\n"
       "                            _aux, _tgt, pos_weight=_pw)")
if old in s and 'aux_pos_weight' not in s.split(old)[0]:
    s = s.replace(old, new, 1)

io.open(TR, 'w', encoding='utf-8', newline='').write(s)
print('aux pos_weight 已加')
