"""支持 IDC_RESET_SCHEDULE=1：加载权重但重置 epoch 计数（否则 1-475<0 → 空循环）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py'
src = open(f, encoding='utf-8-sig').read()

old = """        agent.load(config.model_path)
        current_epoch = agent.globe_eps
        epochs -= current_epoch"""
new = """        agent.load(config.model_path)
        current_epoch = agent.globe_eps
        import os as _os
        if int(_os.environ.get('IDC_RESET_SCHEDULE', '0')) >= 1:
            # 只加载权重、不从旧 epoch 续训（否则 epochs 可能 <= 0，循环直接为空）
            print(f'[TRACE] 重置调度：旧 globe_eps={current_epoch} → 0，'
                  f'epochs={config.epochs}', flush=True)
            current_epoch = 0
            epochs = config.epochs
        else:
            epochs -= current_epoch"""
assert old in src, '找不到 load 后的 epoch 逻辑'
src = src.replace(old, new, 1)

open(f, 'w', encoding='utf-8').write(src.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK', 'IDC_RESET_SCHEDULE' in open(f, encoding='utf-8').read())
