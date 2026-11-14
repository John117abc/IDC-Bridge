# tools 目录说明

一次性排障 / 复现工具，不参与训练主流程。

## 自检与检查

| 脚本 | 用途 |
|---|---|
| `selftest_idc.py` | 35 项断言的离线自检（控制代价、有效性 mask、道路 penalty、椭圆域、完整 66 维管线、IS 权重、GEP 调度、刚体变换不变性、归一化、投影回归）。**改完代码先跑它** |
| `check_config.py` | 打印合并后的完整配置，确认 CLI/YAML 覆盖生效 |
| `stat_state.py` | 统计状态各维的均值/标准差，用于判断归一化与量纲是否合理 |
| `check_wheelbase.py` | 实测 `size.length` 与轴距的对应关系 |

## 训练 / 运行

| 脚本 | 用途 |
|---|---|
| `overnight.sh` | 无人值守训练看护：进程异常退出后按 **episode 编号** 取最新 checkpoint 自动续训（最多 30 次），参数在脚本头部（`TOTAL` / `INTERVAL` / `RATIO` / `SAVE`） |

## 可视化

| 脚本 | 用途 |
|---|---|
| `render_trace.py` | 从 eval 落盘的 `trace_*.npz` 离线渲染 storyboard PNG + GIF（不含凭据，纯本地） |

## 取数与画图

| 脚本 | 用途 |
|---|---|
| `fetch_remote.py` | 从服务器批量下载日志 / GIF / trace。**凭据走环境变量**：`IDC_SSH_HOST` / `IDC_SSH_PORT` / `IDC_SSH_USER` / `IDC_SSH_PASS`（或 `IDC_SSH_KEY`） |
| `plot_train_curves.py` | 解析训练日志，输出 PDMS / 完成度 / 碰撞 / 损失 / 跟踪质量四联图与分段均值表 |

示例：

```bash
export IDC_SSH_HOST=... IDC_SSH_USER=... IDC_SSH_PASS=...
python tools/fetch_remote.py /mnt/2T_HDD/WaymoDatabas/logs/overnight.log -o logs/
python tools/plot_train_curves.py --log logs/overnight.log --out plots/
python tools/fetch_remote.py /mnt/2T_HDD/WaymoDatabas/gifs/20261114 -o plots/gifs --ext .gif,.png
```

## 边界行为记录

| 脚本 | 结论 |
|---|---|
| `test_swap.py` | 换到**另一批**场景时 `set_maps` 可能 abort（`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`） |
| `test_swap_same.py` | 换到**同一批 / 重复批**正常 → 说明 `set_maps` 本身可用，问题在"换到特定场景批" |
| `test_second_env.py` | 同进程内建第二个 env 会 abort（`setCudaHeapSize`）→ madrona 引擎是**进程级单例** |

这些结论直接决定了 `WorldManager.resample()` 的设计（候选池必须取自全量均匀池，不能用最稠密的那批），详见 README「场景管理」一节。
