"""诊断：训练时参考索引是否推进（_ref_idx / step_counter / ref_tensor）。"""
import ast

f = '/gpudrive/baselines/IDC-Bridge/src/scripts/train/idc-train.py'
s = open(f, encoding='utf-8-sig').read()

anchor = """            wm.filter_per_step(states, step)"""
add = """            # --- 诊断：参考索引推进情况（前 5 步） ---
            if step < 5 and getattr(config, 'planner_runtime', ''):
                for _w in range(min(2, config.num_worlds)):
                    _ri = getattr(builder, '_ref_idx', {}).get(_w, -1)
                    _sc = builder.step_counter[_w] if hasattr(builder, 'step_counter') else -1
                    _p0 = None
                    try:
                        _p0 = builder.candidate_paths[_w][int(ego_indices[_w])][0]['pos'][:2]
                    except Exception:
                        pass
                    print(f'[DBG-ref] step={step} w={_w} _ref_idx={_ri} '
                          f'step_counter={_sc} path0={None if _p0 is None else _p0.tolist()}',
                          flush=True)
            wm.filter_per_step(states, step)"""
if '[DBG-ref]' not in s:
    assert anchor in s, '找不到 filter_per_step 锚点'
    s = s.replace(anchor, add, 1)
    open(f, 'w', encoding='utf-8').write(s.lstrip('\ufeff'))
    ast.parse(open(f, encoding='utf-8').read())
    print('DIAG PATCH OK')
else:
    print('已存在诊断')
