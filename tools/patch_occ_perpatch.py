"""把 occ aux 从"全局平均→16×16"改成"逐 token 预测自己的 4×4 patch"

原因：当前实现 `occ_aux_head(occ_tok.mean(1))` 只让头看全局平均，
token 本身不需要携带空间信息 → 训了 3 epoch 探针 Δ 仍只有 1.5e-5（基线 8e-6）。

新实现：每个 occ token（对应 8×8 格 patch）预测自己的 4×4 占据 →
    head: Linear(d, 16) 作用在每个 token 上 → (B,64,16)
    target: occ 任意通道 → maxpool(2) → (B,32,32) → 切 8×8 patch × 4×4
"""
import io

TP = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/train_planner.py'
TR = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/tools/train_plant2_occ.py'

# ---------- 模型：换成逐 token 4×4 ----------
s = io.open(TP, encoding='utf-8', errors='replace').read()
s = s.replace(
    "            self.occ_aux_head = nn.Sequential(\n"
    "                nn.Linear(d, 256), nn.ReLU(inplace=True),\n"
    "                nn.Linear(256, 16 * 16))",
    "            self.occ_aux_head = nn.Sequential(\n"
    "                nn.Linear(d, 128), nn.ReLU(inplace=True),\n"
    "                nn.Linear(128, 4 * 4))", 1)

s = s.replace(
    "            if getattr(self, 'occ_aux', False):\n"
    "                self.last_aux = self.occ_aux_head(\n"
    "                    occ_tok.mean(1)).view(B, 16, 16)",
    "            if getattr(self, 'occ_aux', False):\n"
    "                # 每个 token 预测自己 8x8 patch 内的 4x4 占据\n"
    "                self.last_aux = self.occ_aux_head(occ_tok).view(B, 64, 16)", 1)
io.open(TP, 'w', encoding='utf-8', newline='').write(s)
print('模型：逐 token 4×4 已改')

# ---------- 损失：对应 target ----------
t = io.open(TR, encoding='utf-8', errors='replace').read()
old = """                    _occ_b = b[5]
                    _tgt = (F.max_pool2d(_occ_b.float(), 4).sum(1) > 0).float()
                    _aux = model.last_aux
                    if _aux.shape[0] == _tgt.shape[0]:
                        _laux = F.binary_cross_entropy_with_logits(_aux, _tgt)
                        loss = loss + args.lam_occ * _laux
                        if cnt % 20 == 0:
                            print('    aux %.4f' % float(_laux), flush=True)"""
new = """                    _occ_b = b[5].float()
                    # (B,3,64,64) → 任意通道 → maxpool2 → (B,32,32)
                    _m = (F.max_pool2d(_occ_b, 2).sum(1) > 0).float()   # (B,32,32)
                    # 切成 8x8 个 patch，每个 patch 4x4
                    _tgt = _m.view(-1, 8, 4, 8, 4).permute(0, 1, 3, 2, 4).reshape(-1, 64, 16)
                    _aux = model.last_aux
                    if _aux.shape == _tgt.shape:
                        _laux = F.binary_cross_entropy_with_logits(_aux, _tgt)
                        loss = loss + args.lam_occ * _laux
                        if cnt % 20 == 0:
                            print('    aux %.4f' % float(_laux), flush=True)"""
assert old in t, 'loss 段未匹配'
t = t.replace(old, new, 1)
io.open(TR, 'w', encoding='utf-8', newline='').write(t)
print('损失：逐 patch target 已改')
