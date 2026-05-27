import os
import copy
import yaml
from types import SimpleNamespace


def _deep_merge(base, override):
    """递归合并两个 dict，override 中的值覆盖 base 中的同名字段"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _flatten_selectively(d):
    """将嵌套 YAML 结构扁平化。

    规则：
      - 第一级 key 丢弃（仅作分组标签）
      - 第二级 key 作为最终 key
      - 第三级及以上扁平化：parent_child

    示例:
      training.horizon → horizon
      agent.noise.std → noise_std
    """
    result = {}
    for section, content in d.items():
        if not isinstance(content, dict):
            result[section] = content
            continue
        for k, v in content.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    result[f'{k}_{kk}'] = vv
            else:
                result[k] = v
    return result


def _load_yaml(path: str) -> dict:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f'Config file not found: {path}')
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_include(config_path: str, cfg: dict) -> dict:
    """递归加载 _base 文件并合并，当前文件的键覆盖 base 中的同名字段"""
    base_name = cfg.pop('_base', None)
    if base_name is None:
        return cfg

    base_dir = os.path.dirname(os.path.abspath(config_path))
    base_path = os.path.join(base_dir, base_name)
    base_cfg = _load_yaml(base_path)
    base_cfg = _resolve_include(base_path, base_cfg)

    return _deep_merge(base_cfg, cfg)


def build_config(config_path: str, cli_overrides: dict = None) -> SimpleNamespace:
    """一站式：加载 YAML → 解析 _base 继承 → 扁平化 → CLI 覆盖 → SimpleNamespace。

    Args:
        config_path: YAML 配置文件路径（如 "configs/train.yaml"）
        cli_overrides: {key: value}，key 对应扁平化后的名称，value=None 时不覆盖

    Returns:
        SimpleNamespace，所有参数可通过 .attr 访问
    """
    cfg = _load_yaml(config_path)
    cfg = _resolve_include(config_path, cfg)
    cfg_flat = _flatten_selectively(cfg)
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                cfg_flat[k] = v
    return SimpleNamespace(**cfg_flat)
