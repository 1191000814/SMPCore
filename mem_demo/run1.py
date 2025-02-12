# 使用python脚本自动运行程序

import subprocess
import argparse
from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument('--command', '-c')
parser.add_argument('--switch', '-s')
args = parser.parse_args()

cmd: str = args.command
s = int(args.switch)
ic(cmd)

def run(cmd0=None):
    try:
        if cmd0 is not None:
            cmd = cmd0
        param = cmd.split(' ')
        M = int(param[3][2:])
        ic(M)

        cmd = cmd[:-1]
        sub_processes = []
        sub_processes.append(subprocess.Popen(f'{cmd}{0}', shell=True))

        for i in range(1, M):
            sub_processes.append(subprocess.Popen(f'{cmd}{i}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        ic('wait for subprocesses complete...')
        while any(sub_process.poll() is None for sub_process in sub_processes):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        for sub_process in sub_processes:
            print(f'subprocess {sub_process.pid} end')
            sub_process.terminate()


if __name__ == '__main__':
    for d in [6, 5, 4, 3, 2, 1]:
        for l in [1, 2, 3]:
            # for s in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
            run(f'python3 -u ~/secure-MLG/mem_demo/smpcore.py -M3 -d{d} -l{l} -v3 -s{s} -I0')
