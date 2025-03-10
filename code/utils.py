import networkx as nx
import numpy as np
from icecream import ic

def get_degree(G: nx.Graph, V):
    '''
    获取单层图的度列表
    '''
    degrees = G.degree
    deg_list = [-1 for _ in range(V)]
    for node_id, degree in degrees:
        # print(node_id, degree)
        # graph node id begin with 0
        deg_list[node_id] = degree
    return deg_list


def get_all_degree(MG: nx.MultiGraph | nx.MultiDiGraph, V, L):
    if isinstance(MG, nx.MultiDiGraph):
        all_deg_list = np.zeros([L, V, 2], dtype=np.int64)
        for u, v, layer in MG.edges:
            all_deg_list[layer, u, 1] += 1
            all_deg_list[layer, v, 0] += 1
    else:
        all_deg_list = np.zeros([L, V], dtype=np.int64)
        for u, v, layer in MG.edges:
            # ic(u, v, layer)
            all_deg_list[layer, v] += 1
            all_deg_list[layer, u] += 1
    return all_deg_list


def get_subgraph(MDG: nx.MultiDiGraph, k, lamb, V, L):
    S = set()
    I_out = [-1 for _ in range(V)]
    Degree = get_all_degree(MDG, V, L)
    for v in MDG.nodes:
        I_out[v] = np.partition(Degree[:, v, 1], -lamb)[-lamb]
        if I_out[v] >= k:
            S.add(v)
    H = nx.MultiDiGraph()

    H.add_nodes_from(MDG.nodes)

    for u, v, layer in MDG.edges:
        if u in S:
            H.add_edge(u, v, layer)

    return H, S, I_out


def get_sub_degree(MDG: nx.MultiDiGraph, S: set, T: set, V, L):
    all_deg_list = np.zeros([L, V, 2], dtype=np.int64)
    for u in S:
        for v in MDG[u]:
            if v in T:
                for layer in MDG[u][v]:
                    all_deg_list[layer][u][1] += 1
                    all_deg_list[layer][v][0] += 1
    return all_deg_list


def get_adj_mat(G: nx.Graph, V):
    '''
    获取单层图的邻接矩阵
    '''
    adj_mat = np.zeros([V, V], dtype=np.int8)
    for u, v in G.edges:
        adj_mat[u][v] = 1
        adj_mat[v][u] = 1
    return adj_mat


if __name__ == '__main__':
    # G = create_layer(1)
    # ic(get_degree(G))
    # MG = create_graph()
    # ic(get_all_degree(MG))
    pass
