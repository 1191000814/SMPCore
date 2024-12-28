from icecream import ic
import numpy as np
import networkx as nx

G = nx.barabasi_albert_graph(100, 2)
ic(G.number_of_edges())
ic(G.number_of_nodes())


# for v in G.edges:
#     ic(v)