# 自己创建数据集
from icecream import ic
import networkx as nx


def create_3layer() -> nx.MultiDiGraph:
    V = 13  # 顶点数
    L = 3  # 层数
    edges = [set() for _ in range(L)]
    # 在每一层加入边(一条边就是一个二元组)
    edges[0].update({(1, 2), (1, 3), (1, 8), (2, 3), (2, 8), (3, 8), (4, 5), (4, 6), (5, 6), (5, 7), (5, 9),
                    (5, 10), (6, 7), (9, 10), (9, 11), (9, 13), (10, 11), (10, 13), (11, 12), (11, 13)})  # 20条边
    edges[1].update({(2, 3), (2, 4), (2, 7), (3, 4), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (
        5, 6), (5, 7), (7, 8), (9, 10), (9, 11), (9, 12), (10, 11), (10, 12), (11, 12), (12, 13)})  # 20条边
    edges[2].update({(1, 2), (1, 3), (1, 8), (2, 3), (2, 4), (2, 7), (2, 8), (3, 4), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (
        4, 8), (5, 6), (5, 7), (6, 7), (7, 8), (9, 10), (9, 10), (9, 11), (9, 12), (9, 13), (11, 12), (11, 13), (12, 13)})  # 25条边
    for l in range(L):
        ic(len(edges[l]))
    E = sum(len(layer) for layer in edges)  # 总共的边数
    ic(E)

    # 创建之前先删除原有节点
    G = nx.MultiDiGraph()
    for l in range(L):
        # 创建节点
        G.add_nodes_from([(l * V + i + 1, {'id': str(i), 'layer': str(l)})
                          for i in range(V)])
        # 创建边
        edge_layer = [(from_v + l * V, to_v + l * V)
                      for from_v, to_v in edges[l]]
        G.add_edges_from(edge_layer)
    ic(G.number_of_nodes())
    ic(G.number_of_edges())
    assert G.number_of_nodes() == V * L
    assert G.number_of_edges() == E
    return G


def create_3layer_1():
    V = 15  # 顶点数
    L = 3  # 层数
    edges = [set() for _ in range(L)]
    # 在每一层加入边(一条边就是一个二元组)
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
    for l in range(L):
        ic(len(edges[l]))
    E = sum(len(layer) for layer in edges)  # 总共的边数
    ic(E)

    # 创建之前先删除原有节点
    G = nx.MultiDiGraph()
    for l in range(L):
        # 创建节点
        G.add_nodes_from([(l * V + i, {'id': str(i), 'layer': str(l)})
                          for i in range(V)])
        # 创建边
        edge_layer = [(from_v + l * V, to_v + l * V)
                      for from_v, to_v in edges[l]]
        G.add_edges_from(edge_layer)
    ic(G.number_of_nodes())
    ic(G.number_of_edges())
    assert G.number_of_nodes() == V * L
    assert G.number_of_edges() == E
    return G
