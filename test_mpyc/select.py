# 比较快速选择和排序

from icecream import ic
from mpyc.runtime import mpc
from mpyc.statistics import _quickselect
import numpy as np
import random
import time

sec_int = mpc.SecInt(16)

ls = np.array([])

ls = [i for i in range(10000)]

random.shuffle(ls)
ls = sec_int.array(np.array(ls).reshape([100, 100]))
sort_ls = [0 for _ in range(100)]

for i in range(100):
    sort_ls[i] = np.sort(ls[i])[10]
    # sort_ls[i] = await mpc.output(np.sort(ls[i])[10])