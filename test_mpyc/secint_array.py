# 使用secint.array
from icecream import ic
from mpyc.runtime import mpc
from mpyc.seclists import seclist
import numpy as np
import random
import time


async def func():

    async with mpc:

        start = time.time()

        sec_int = mpc.SecInt(16)

        ls = [i for i in range(10000)]
        random.shuffle(ls)
        ls = sec_int.array(np.array(ls).reshape([100, 100]))
        sort_ls = [0 for _ in range(100)]

        for i in range(100):
            sort_ls[i] = np.sort(ls[i])[10]
            # sort_ls[i] = await mpc.output(np.sort(ls[i])[10])

        sort_ls = await mpc.output(sort_ls)

        # ic(sort_ls)

        # ic(ls[0])
        # ic(ls[0].shape)
        # mpc.sorted(ls)

        ic(time.time() - start)


mpc.run(func())
