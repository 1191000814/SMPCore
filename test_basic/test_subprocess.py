# 测试一下子进程

import subprocess
import time
from icecream import ic

cmd = 'ls'

subprocess.Popen(f'{cmd}', shell=True)

for i in range(3):
    subprocess.Popen(f'{cmd}', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
ic(1)