# 自己创建数据集
from icecream import ic
import networkx as nx
import numpy as np

# 构建多层图的一层


def create_graph_i(l) -> nx.MultiDiGraph:
    V = 15  # 顶点数
    L = 3  # 层数
    edges = [set() for _ in range(L)]
    edges[0].update({
        (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6,
                                                 8), (6, 9), (7, 8), (7, 9), (8, 9),
        (10, 11), (10, 12), (10, 13), (10, 14), (11,
                                                 12), (11, 13), (11, 14), (12, 13), (13, 14)
    })
    edges[1].update({
        (0, 1), (0, 3), (0, 4), (1, 3), (1, 4), (3, 4),
        (5, 6), (5, 7), (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9),
        (10, 11), (10, 12), (10, 13), (10, 14), (11,
                                                 12), (11, 13), (11, 14), (12, 13), (13, 14)
    })
    edges[2].update({
        (0, 1), (0, 2), (0, 4), (1, 2), (1, 4), (2, 4),
        (5, 8), (5, 9), (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
        (10, 11), (10, 12), (10, 13), (10, 14), (11,
                                                 12), (11, 14), (12, 13), (12, 14), (13, 14)
    })
    edge = edges[l]  # 该层的边
    E = len(edge)  # 总共的边数
    ic(E)

    # 创建之前先删除原有节点
    G = nx.MultiDiGraph()
    # 创建节点, 图节点标识是从1开始的, id是从0开始的
    G.add_nodes_from([(i, {'id': str(i)}) for i in range(V)])
    # 创建边
    G.add_edges_from(edge)
    ic(G.number_of_nodes())
    ic(G.number_of_edges())
    assert G.number_of_nodes() == V
    assert G.number_of_edges() == E
    return G


def get_degree(G: nx.MultiDiGraph):
    # 获取度列表
    V = 15  # 顶点数
    L = 3  # 层数
    degrees = G.degree()
    deg_list = [-1 for _ in range(V)]
    for node_id, degree in degrees:
        # print(node_id, degree)
        # 图节点标识是从1开始的
        deg_list[node_id - 1] = degree
    return deg_list


def get_adj_mat(G: nx.MultiDiGraph):
    # 获取邻接矩阵
    V = 15  # 顶点数
    L = 3  # 层数
    adj_mat = np.zeros([V, V], dtype=np.int8)
    for u, v, k in G.edges:
        # 无向图
        adj_mat[u][v] = 1
        adj_mat[v][u] = 1
    return adj_mat


G = create_graph_i(1)
ic(get_degree(G))
G.remove_node(2)
ic(get_degree(G))
