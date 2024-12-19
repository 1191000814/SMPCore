import networkx as nx
from icecream import ic

MG = nx.MultiGraph()
MG.add_nodes_from([1, 2])
MG.add_edge(1, 2, 1)
MG.add_edge(2, 1, 1)
ic(MG.edges)

# G:nx.Graph = nx.erdos_renyi_graph(1000, 0.5)
# ic(G.number_of_edges())
