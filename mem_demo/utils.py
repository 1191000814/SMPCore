import networkx as nx
import numpy as np

# from icecream import ic


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
    '''
    获取整个多层图的度列表
    '''
    # 即使节点被删除, 但是邻接矩阵的形状不会变
    if isinstance(MG, nx.MultiDiGraph):
        # 有向图
        all_deg_list = np.zeros([L, V, 2], dtype=np.int64)  # 入度, 出度
        for u, v, layer in MG.edges:
            all_deg_list[layer, u, 1] += 1  # u的出度
            all_deg_list[layer, v, 0] += 1  # v的入度
    else:
        # 无向图
        all_deg_list = np.zeros([L, V], dtype=np.int64)
        for u, v, layer in MG.edges:
            # ic(u, v, layer)
            all_deg_list[layer, v] += 1
            all_deg_list[layer, u] += 1
    return all_deg_list


def get_subgraph(MDG: nx.MultiDiGraph, k, lamb, V, L):
    '''
    获取MDG一个子图, 该子图是一个[S:T]诱导子图,
    其中T已经给出, S是MDG中(Top-λ)出度>=k的节点
    '''
    S = set()
    I_out = [-1 for _ in range(V)]
    Degree = get_all_degree(MDG, V, L)
    for v in MDG.nodes:
        # S只包含第λ大的out度>=k的节点
        I_out[v] = np.partition(Degree[:, v, 1], -lamb)[-lamb]
        if I_out[v] >= k:
            S.add(v)
    # 创建一个新的MultiDiGraph来存储结果
    H = nx.MultiDiGraph()

    # 添加S中的所有顶点到新图中（即使它们没有边指向T）
    H.add_nodes_from(MDG.nodes)

    # 遍历G中所有的边
    for u, v, layer in MDG.edges:
        # 检查边的起点是否在S中，终点是否在T中
        if u in S:
            # 如果满足条件，则在新图中添加这条边和它的数据
            H.add_edge(u, v, layer)

    return H, S, I_out


def get_sub_degree(MDG: nx.MultiDiGraph, S: set, T: set, V, L):
    '''
    获取整个多层图MDG在导出子图H=[S:T]中的的度列表
    '''
    all_deg_list = np.zeros([L, V, 2], dtype=np.int64)  # 入度, 出度
    for u in S:
        for v in MDG[u]:
            if v in T:
                for layer in MDG[u][v]:
                    all_deg_list[layer][u][1] += 1  # u的出度
                    all_deg_list[layer][v][0] += 1  # v的入度
    return all_deg_list


def get_adj_mat(G: nx.Graph, V):
    '''
    获取单层图的邻接矩阵
    '''
    adj_mat = np.zeros([V, V], dtype=np.int8)
    for u, v in G.edges:
        # 无向图
        adj_mat[u][v] = 1
        adj_mat[v][u] = 1
    return adj_mat


if __name__ == '__main__':
    # G = create_layer(1)
    # ic(get_degree(G))
    # MG = create_graph()
    # ic(get_all_degree(MG))
    pass
