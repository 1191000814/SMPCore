from icecream import ic
import numpy as np
from mpyc.runtime import mpc
from mpyc.seclists import seclist

ls = [1, 2, 3]
ls1 = ls.copy()
ls1[1] = 10
ic(ls)
ic(ls1)