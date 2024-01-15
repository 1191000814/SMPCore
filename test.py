from icecream import ic
import numpy as np

a = [[1, 2, 3], [2, 3, 4]]
a = np.array(a).T
ic(a.tolist())