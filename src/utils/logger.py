# utils/logger.py

import os
import logging
from datetime import datetime
from pathlib import Path
from utils.common import get_project_root

file_dir = get_project_root() / 'logs'


def _pick_writable_dir(candidates):
    """返回第一个可创建且可写的日志根目录；全部不可写则返回 None。

    原来日志目录硬编码为 '/workspace/data/logs'（docker 容器内路径），
    在主机上直接跑会 PermissionError / FileNotFoundError。这里做有序回退，
    可用环境变量 IDC_LOG_DIR 覆盖。
    """
    for c in candidates:
        if not c:
            continue
        try:
            dated = Path(c) / datetime.now().strftime("%Y%m%d")
            os.makedirs(dated, exist_ok=True)
            probe = dated / ".write_test"
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write("")
            probe.unlink()
            return Path(c)
        except OSError:
            continue
    return None


def get_logger(
        name: str = __name__,
        log_dir: str = None,
        level: int = logging.INFO,
        use_console: bool = True,
        use_file: bool = True
):
    """
    获取 logger，将同一天、同一模块的日志写入同一个文件。

    日志文件路径示例：logs/20260112/train.log

    Args:
        name: logger 名称（如 'train', 'eval'）
        log_dir: 基础日志目录；None 时按 IDC_LOG_DIR > /workspace/data/logs
                 > <项目根>/logs > ~/idc_logs 依次尝试，取第一个可写的
        level: 日志级别
        use_console: 是否输出到控制台
        use_file: 是否写入文件

    Returns:
        logging.Logger 实例
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已配置，直接返回

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- 控制台输出 ---
    if use_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # --- 文件输出：按天 + 模块 聚合 ---
    if use_file:
        # 1. 选一个可写的日志根目录
        base_log_dir = _pick_writable_dir([
            log_dir,
            os.environ.get('IDC_LOG_DIR'),
            '/workspace/data/logs',
            str(get_project_root() / 'logs'),
            os.path.join(os.path.expanduser('~'), 'idc_logs'),
        ])
        if base_log_dir is None:
            logger.warning('[Logger] 找不到可写日志目录，本次仅输出到控制台')
            return logger

        today_str = datetime.now().strftime("%Y%m%d")
        dated_log_dir = base_log_dir / today_str

        # 2. 文件名：模块名.log （如 train.log）
        clean_name = name.split('.')[-1] if '.' in name else name
        log_filename = f"{clean_name}.log"
        log_filepath = dated_log_dir / log_filename

        # 3. 使用 FileHandler（默认 mode='a' 追加写入）
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 可选：首次创建时提示
        if not hasattr(logger, '_file_logged'):
            print(f"[Logger] 日志追加到: {log_filepath}")
            logger._file_logged = True

    return logger