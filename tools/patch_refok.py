"""在 maybe_replan 里打印“规划轨迹到最近车道段”的距离，判断参考是否越界。"""
import ast

pr = '/gpudrive/baselines/IDC-Bridge/src/env/planner_runtime.py'
s = open(pr, encoding='utf-8-sig').read()

anchor = """        return None"""
# 用一个更稳的锚点：maybe_replan 结束前的 return（找到 def maybe_replan 后的最后一个 return None）
i = s.find('    def maybe_replan(')
j = s.find('    def ', i + 10)
block = s[i:j]
assert 'return None' in block, 'maybe_replan 内没有 return None'

diag = '''        # ---- 诊断：参考是否在车道内（V1）----
        if self.debug:
            try:
                for w in range(builder.num_worlds):
                    pk = self.packs.get(w)
                    P = paths.get(w)
                    if pk is None or P is None or len(getattr(pk, 'seg_a', [])) == 0:
                        continue
                    pts = np.asarray(P[0][:41], np.float32)
                    A = pk.seg_a
                    d = np.full(len(pts), 1e9, np.float32)
                    for k, q in enumerate(pts):
                        d[k] = np.linalg.norm(A - q[None], axis=1).min()
                    print('[RT][ref-ok] w=%d 到最近车道: 中位 %.2f 最大 %.2f '
                          '越界(>3.5m) 占比 %.0f%%'
                          % (w, float(np.median(d)), float(d.max()),
                             100.0 * float((d > 3.5).mean())), flush=True)
            except Exception as _e:
                print(f'[RT][ref-ok] 诊断失败: {_e}', flush=True)
        return None'''
block_new = block.replace('        return None', diag, 1)
s = s[:i] + block_new + s[j:]
open(pr, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
ast.parse(open(pr, encoding='utf-8').read())
print('PATCH OK')
