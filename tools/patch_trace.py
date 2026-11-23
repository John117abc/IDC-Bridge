"""在训练循环入口加 trace，定位静默退出点。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py'
src = open(f, encoding='utf-8-sig').read()
n = 0

old = """    for epoch in range(epochs):
        epoch += current_epoch
        obs = env.reset()"""
new = """    print('[TRACE] 进入 epoch 循环', flush=True)
    for epoch in range(epochs):
        epoch += current_epoch
        print(f'[TRACE] epoch={epoch} 即将 env.reset()', flush=True)
        obs = env.reset()
        print('[TRACE] env.reset() 完成', flush=True)"""
if '[TRACE] 进入 epoch 循环' not in src:
    assert old in src, '找不到 epoch 循环锚点'
    src = src.replace(old, new, 1)
    n += 1

open(f, 'w', encoding='utf-8').write(src.lstrip('\ufeff'))
ast.parse(open(f, encoding='utf-8').read())
print('PATCH OK 处数 =', n)
