BASE = '/home/jiangchengxuan/PycharmProjects/gpudrive/baselines/IDC-Bridge/src/env/'
for path, pats in [
    (BASE + 'candidate_sampler.py',
     ['speed', 'road_dist', "dict(", 'clearance', 'def speed_profile']),
    (BASE + 'idc_state_builder.py',
     ['spd_ahead', 'curv_ahead', 'road_ahead', 'ref_error', 'ref_tensor', 'num_candidate']),
]:
    print('=' * 20, path.split('/')[-1])
    for i, line in enumerate(open(path), 1):
        if any(p in line for p in pats):
            print(f'{i:4d}: {line.rstrip()[:110]}')
