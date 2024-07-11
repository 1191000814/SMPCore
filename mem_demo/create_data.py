from icecream import ic
import networkx as nx
import os
import sys

V = 15  # num of v
L = 3  # num of l
edges = [set() for _ in range(L)]
edges[0].update(
    {
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (5, 8),
        (5, 9),
        (6, 7),
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
        (8, 9),
        (10, 11),
        (10, 12),
        (10, 13),
        (10, 14),
        (11, 12),
        (11, 13),
        (11, 14),
        (12, 13),
        (13, 14),
    }
)

edges[1].update(
    {
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 3),
        (1, 4),
        (3, 4),
        (5, 6),
        (5, 7),
        (5, 8),
        (5, 9),
        (6, 7),
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
        (10, 11),
        (10, 12),
        (10, 13),
        (10, 14),
        (11, 12),
        (11, 13),
        (11, 14),
        (12, 13),
        (13, 14),
    }
)

edges[2].update(
    {
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 2),
        (1, 4),
        (2, 4),
        (5, 8),
        (5, 9),
        (6, 7),
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
        (8, 9),
        (10, 11),
        (10, 12),
        (10, 13),
        (10, 14),
        (11, 12),
        (11, 14),
        (12, 13),
        (12, 14),
        (13, 14),
    }
)


def create_layer(l, di=False):
    '''
    create a layer of graph
    '''
    edge = edges[l % L]  # edges in this layer
    E = len(edge)  # edge num in this layer
    ic(E)
    if di:  # directed graph
        G = nx.DiGraph()
    else:  # undirected graph
        G = nx.Graph()
    # create nodes
    # the id of node begin with 0 instead of 1
    G.add_nodes_from([(i, {'id': str(i)}) for i in range(V)])
    # create edges
    G.add_edges_from(edge)
    ic(G.number_of_nodes())
    ic(G.number_of_edges())
    assert G.number_of_nodes() == V
    assert G.number_of_edges() == E
    return G


def create_graph(di=False):
    '''
    create total graph
    '''
    # E = sum([len(edges[l])] for l in range(L))
    E = len(edges[0]) + len(edges[1]) + len(edges[2])
    # ic(E)
    if di:
        MG = nx.MultiDiGraph()
    else:
        MG = nx.MultiGraph()
    # create nodes
    # the id of node begin with 0 instead of 1
    MG.add_nodes_from([(i, {'id': str(i)}) for i in range(V)])
    # create edge
    for layer, edge in enumerate(edges):
        # each layer of graph
        MG.add_edges_from([u, v, layer] for u, v in edge)
    ic(MG.number_of_nodes())
    ic(MG.number_of_edges())
    assert MG.number_of_nodes() == V
    assert MG.number_of_edges() == E
    return MG


def create_by_file(name: str, di=False):
    '''
    从文件路径创建多层图
    '''
    if name is None:
        return create_graph(di), 3
    if di:  # 有向图
        MG = nx.MultiDiGraph()
    else:  # 无向图
        MG = nx.MultiGraph()
    nodes = set()
    edges = []
    layers = set()
    # 数据中的layer都是从1开始的
    num_layers = 0

    project_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(project_dir, 'data', f'{name}.txt')
    ic(path)
    with open(path, 'r', encoding='utf-8') as file:
        next(file)  # 第一行是 层数/节点数
        for line in file:
            l, u, v = line.split()
            # 源节点,目标节点,层数
            # ? 节点, 层数都不一定是按照0-~num_v排列的
            u, v, l = int(u), int(v), int(l)
            nodes.add(u)
            nodes.add(v)
            edges.append([u, v, l])
            layers.add(l)
            num_layers = max(l + 1, num_layers)
        nodes = sorted(list(nodes))
        layers = sorted(list(layers))
        nodes_map = {node: i for i, node in enumerate(nodes)}
        layers_map = {layer: i for i, layer in enumerate(layers)}
        MG.add_nodes_from(nodes_map.values())
        MG.add_edges_from([(nodes_map[u], nodes_map[v], layers_map[layer]) for u, v, layer in edges])
    ic(MG.number_of_nodes())
    ic(MG.number_of_edges())
    ic(num_layers)
    return MG, num_layers


def create_layer_by_file(name: str, layer, di=False):
    '''
    从文件路径创建多层图的第layer层, layer从0开始计数
    '''
    if di:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    nodes = set()
    edges = []
    layers = set()
    num_layers = 0
    #! 必须使用相对路径, 以免后面造成很多麻烦
    project_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(project_dir, 'data', f'{name}.txt')
    ic(path)
    # path = os.path.expanduser(f'~/Secure-Graph/data/{name}.txt')
    # path = f'../data/{name}.txt'
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            l, u, v = line.split()
            # 源节点,目标节点,层数
            # ? 节点, 层数都不一定是按照0-~num_v排列的
            u, v, l = int(u), int(v), int(l)
            nodes.add(u)
            nodes.add(v)
            edges.append([u, v, l])
            layers.add(l)
            num_layers = max(l + 1, num_layers)
            assert layer < num_layers
        # ? 重点: 每个单层图的顶点id都保持一致(和多层图的顶点id一致)
        nodes = sorted(list(nodes))
        layers = sorted(list(layers))
        nodes_map = {node: i for i, node in enumerate(nodes)}
        layers_map = {layer: i for i, layer in enumerate(layers)}
        G.add_nodes_from(nodes_map.values())
        for u, v, l in edges:
            if layers_map[l] == layer:
                G.add_edges_from([(nodes_map[u], nodes_map[v])])
    ic(G.number_of_nodes())
    ic(G.number_of_edges())
    return G
