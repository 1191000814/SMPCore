import networkx as nx
from icecream import ic
import create_data
import numpy as np
import collections
import os
import utils

# MG, n = create_data.create_by_file(None)
# # G = create_data.create_layer_by_file('homo', 0)
# ic(set(MG.neighbors(1)))

s = set()
s.add(1)
s=  s.union({2, 3})
ic(s)