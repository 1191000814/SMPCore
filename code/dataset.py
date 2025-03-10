from icecream import ic
import networkx as nx
import random
import os
import numpy as np

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


def read_my_layer(l, di=False):
    '''
    读取示例图的的第l层
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


def read_my_graph(di=False):
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


def read_layer2(name: str, di=False):
    if name is None:
        return create_graph(di), 3
    if di:
        MG = nx.MultiDiGraph()
    else:
        MG = nx.MultiGraph()
    nodes = set()
    edges = []
    layers = set()
    num_layers = 0

    project_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(project_dir, 'data', f'{name}.txt')
    ic(path)
    with open(path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            l, u, v = line.split()
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


def read_graph(name: str, get_num_layer=None, di=False):
    if name is None:
        return create_graph(di), 3
    if di:  # 有向图
        MG = nx.MultiDiGraph()
    else:  # 无向图
        MG = nx.MultiGraph()
    edges = []
    num_layers = 0
    num_nodes = 0
    project_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(project_dir, 'data', f'{name}.txt')
    ic(path)
    with open(path, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            l, u, v = line.split()
            u, v, l = int(u) - 1, int(v) - 1, int(l) - 1
            num_layers = max(l + 1, num_layers)
            num_nodes = max(u + 1, v + 1, num_nodes)
            edges.append([u, v, l])
        if get_num_layer is None:
            MG.add_nodes_from([node_id for node_id in range(num_nodes)])
            MG.add_edges_from([(u, v, l) for u, v, l in edges])
        else:
            if get_num_layer <= num_layers:
                MG.add_nodes_from([node_id for node_id in range(num_nodes)])
                MG.add_edges_from([(u, v, l) for u, v, l in edges if l < get_num_layer])
            else:
                MG.add_nodes_from([node_id for node_id in range(num_nodes)])
                MG.add_edges_from([(u, v, l) for u, v, l in edges])
                get_num_layer = num_layers

    ic(MG.number_of_nodes())
    ic(MG.number_of_edges())
    ic(num_layers)
    return MG, num_layers


def read_layer(name: str, layer, di=False):
    if di:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    nodes = set()
    edges = []
    layers = set()
    num_layers = 0
    project_dir = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(project_dir, 'data', f'{name}.txt')
    ic(path)
    with open(path, 'r', encoding='utf-8') as file:
        line0 = file.readline()
        num_layers, _, _ = line0.split()
        if layer >= int(num_layers):
            layer %= int(num_layers)
        for line in file:
            l, u, v = line.split()
            u, v, l = int(u), int(v), int(l)
            nodes.add(u)
            nodes.add(v)
            edges.append([u, v, l])
            layers.add(l)
            # num_layers = max(l + 1, num_layers)
            # assert layer < num_layers
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


def write_graph(MG: nx.MultiGraph, num_layers):
    M = 100
    project_dir = os.path.dirname(os.path.dirname(__file__))
    for i in range(1, M + 1):
        path = os.path.join(project_dir, 'data', 'synthetic', f'random_{i}.txt')
        if not os.path.exists(path):
            file_name = f'random_{i}'
            ic(path)
            break
        elif i == M:
            return
    num_nodes = MG.number_of_nodes()
    edges = sorted(MG.edges(keys=True), key=lambda edge: (edge[2], edge[0], edge[1]))

    with open(path, 'w') as f:
        f.write(f'{num_layers} {num_nodes} {num_nodes}\n')
        for from_v, to_v, layer in edges:
            f.write(f'{layer + 1} {from_v + 1} {to_v + 1}\n')
    return file_name


def gen_random_graph(num_nodes, num_layers, aver_deg):
    num_edges = int(num_layers * num_nodes * aver_deg)
    max_layer_edges = num_nodes * (num_nodes - 1) // 2
    max_total_edges = num_layers * max_layer_edges
    assert num_edges <= max_total_edges
    MG = nx.MultiGraph()
    MG.add_nodes_from(range(num_nodes))
    edges_idx = random.sample(range(max_total_edges), num_edges)
    for edge_idx in edges_idx:
        layer = edge_idx // max_layer_edges  # 层数
        from_v, to_v = kth_position(edge_idx % max_layer_edges, num_nodes)
        MG.add_edge(from_v, to_v, layer)
    return MG


def kth_position(k, n):
    assert k >= 0 and k < n * (n - 1)
    total = 0
    for i in range(n - 1):
        row_count = n - i - 1
        if total + row_count >= k:
            j = i + (k - total)
            return i, j
        total += row_count
    return None


def gen_random_skew_graph(num_nodes, num_layers, aver_deg, skew_list):
    num_edges = int(num_layers * num_nodes * aver_deg)
    max_layer_edges = num_nodes * (num_nodes - 1) // 2
    max_total_edges = num_layers * max_layer_edges
    assert num_edges <= max_total_edges
    assert num_layers == len(skew_list)
    skew_sum = sum(skew_list)
    skew_list = [skew / skew_sum for skew in skew_list]
    num_edges_per_layer = [int(skew * num_edges) for skew in skew_list]
    num_edges_per_layer[num_layers - 1] = num_edges - sum(num_edges_per_layer[:-1])
    ic(num_edges_per_layer)
    MG = nx.MultiGraph()
    MG.add_nodes_from(range(num_nodes))
    for layer in range(num_layers):
        num_edges = num_edges_per_layer[layer]
        edges_idx = random.sample(range(max_layer_edges), num_edges)
        for edge_idx in edges_idx:
            from_v, to_v = kth_position(edge_idx % max_layer_edges, num_nodes)
            MG.add_edge(from_v, to_v, layer)
    return MG


def gen_BA_graph(num_nodes, num_layers, num_connect):
    MG = nx.MultiGraph()
    MG.add_nodes_from(range(num_nodes))
    for layer in range(num_layers):
        G = nx.barabasi_albert_graph(num_nodes, num_connect)
        edges = ((from_v, to_v, layer) for from_v, to_v in G.edges)
        MG.add_edges_from(edges)
    return MG


if __name__ == '__main__':
    dense_ls = [0.25, 0.5, 1, 2, 4, 8, 16, 32]
    size_ls = [1000, 2000, 4000, 6000, 8000, 10000, 15000, 20000]
    skew_ls = [[1, 1, 1], [1, 2, 3], [1, 3, 5], [1, 4, 7], [1, 5, 9], [1, 6, 11], [1, 7, 13], [1, 8, 15]]
    for skew in skew_ls:
        MG = gen_random_skew_graph(10000, 3, 4, skew)
        ic(MG.number_of_nodes())
        ic(MG.number_of_edges())
        write_graph(MG, 3)
