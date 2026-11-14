"""从远端服务器批量下载产物（日志 / GIF / trace npz）。

凭据从环境变量读取，不写进仓库：
    IDC_SSH_HOST   主机
    IDC_SSH_PORT   端口（默认 22）
    IDC_SSH_USER   用户名
    IDC_SSH_PASS   密码（或用 IDC_SSH_KEY 指定私钥路径）

用法：
    # 下载单个文件
    python tools/fetch_remote.py /mnt/2T_HDD/WaymoDatabas/logs/overnight.log -o logs/

    # 下载某目录下匹配后缀的全部文件
    python tools/fetch_remote.py /mnt/2T_HDD/WaymoDatabas/gifs/20261114 -o plots/gifs --ext .gif,.png

    # 只列目录
    python tools/fetch_remote.py /mnt/2T_HDD/WaymoDatabas/gifs --list
"""
import argparse
import os
import sys

import paramiko


def connect():
    host = os.environ.get("IDC_SSH_HOST")
    user = os.environ.get("IDC_SSH_USER")
    if not host or not user:
        raise SystemExit("请先设置 IDC_SSH_HOST / IDC_SSH_USER / IDC_SSH_PASS")
    port = int(os.environ.get("IDC_SSH_PORT", "22"))
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key = os.environ.get("IDC_SSH_KEY")
    if key:
        cli.connect(host, port=port, username=user, key_filename=key, timeout=20)
    else:
        pwd = os.environ.get("IDC_SSH_PASS")
        if not pwd:
            raise SystemExit("请设置 IDC_SSH_PASS 或 IDC_SSH_KEY")
        cli.connect(host, port=port, username=user, password=pwd, timeout=20)
    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("remote", help="远端文件或目录")
    ap.add_argument("-o", "--out", default=".", help="本地输出目录")
    ap.add_argument("--ext", default="", help="逗号分隔的后缀过滤，如 .gif,.png")
    ap.add_argument("--list", action="store_true", help="只列目录不下")
    args = ap.parse_args()

    cli = connect()
    sftp = cli.open_sftp()
    try:
        st = sftp.stat(args.remote)
        is_dir = (st.st_mode & 0o040000) != 0
        os.makedirs(args.out, exist_ok=True)
        if not is_dir:
            dst = os.path.join(args.out, os.path.basename(args.remote))
            sftp.get(args.remote, dst)
            print(f"{args.remote} -> {dst} ({os.path.getsize(dst)} bytes)")
            return
        names = sorted(sftp.listdir(args.remote))
        exts = [e.strip() for e in args.ext.split(",") if e.strip()]
        if exts:
            names = [n for n in names if any(n.endswith(e) for e in exts)]
        if args.list:
            for n in names:
                print(n)
            print(f"-- {len(names)} item(s)")
            return
        for n in names:
            rp = f"{args.remote}/{n}"
            try:
                if (sftp.stat(rp).st_mode & 0o040000) != 0:
                    continue
            except IOError:
                continue
            dst = os.path.join(args.out, n)
            sftp.get(rp, dst)
            print(f"{n} ({os.path.getsize(dst)} bytes)")
        print(f"-- {len(names)} item(s)")
    finally:
        sftp.close()
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
