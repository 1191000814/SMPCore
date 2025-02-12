from icecream import ic
import numpy as np
import networkx as nx
import random

G = nx.barabasi_albert_graph(10, 2)
ic(G.number_of_edges())
ic(G.number_of_nodes())

# edges = G.edges(data=True)
# edges = sorted(edges, key=lambda edge: (edge[0], edge[1]))

ls = random.sample(range(10), 3)
ic(ls)
ic(ls + 1)