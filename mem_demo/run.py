# 使用python脚本自动运行程序

import subprocess
import argparse
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--command', '-c')
args = parser.parse_args()

cmd: str = args.command
ic(cmd)

sub_processes = []

try:
    param = cmd.split(' ')
    M = int(param[3][2:])
    ic(M)

    cmd = cmd[:-1]
    sub_processes.append(subprocess.Popen(f'{cmd}{0}', shell=True))

    for i in range(1, M):
        sub_processes.append(subprocess.Popen(f'{cmd}{i}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    ic('wait for subprocesses complete...')
    while any(sub_process.poll() is None for sub_process in sub_processes):
        pass
except KeyboardInterrupt:
    for sub_process in sub_processes:
        ic(f'kill subprocess {sub_process.pid}')
        sub_process.terminate()
