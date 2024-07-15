import networkx as nx
from icecream import ic

MG = nx.MultiGraph()
MG.add_nodes_from([1, 2])
MG.add_edge(1, 2, 1)
MG.add_edge(1, 2, 0)
ic(MG.edges)