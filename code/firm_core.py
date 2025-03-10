"""
firmcore decomposition of multilayer networks
"""

import utils
import numpy as np
from icecream import ic
from tqdm import tqdm
import networkx as nx
import time
import collections
import dataset as DS


def firm_core1(MG0: nx.MultiGraph, num_layers, lamb):
    MG = MG0.copy()
    start_time = time.time()
    num_nodes = MG.number_of_nodes()
    assert lamb <= num_layers
    I = [-1 for _ in range(num_nodes)]
    B = collections.defaultdict(set)
    core = collections.defaultdict(set)
    Degree = utils.get_all_degree(MG, num_nodes, num_layers)
    with tqdm(range(num_nodes)) as pbar:
        for i in range(num_nodes):
            top_d = np.partition(Degree[:, i], -lamb)[-lamb]
            assert top_d <= num_nodes
            I[i] = top_d
            B[top_d].add(i)
        # ic(I)
        # ic(B)
        count = 0
        for k in range(num_nodes - 1):
            if count == num_nodes:
                break
            # ic(f'-------------{k}--------------')
            while B[k]:
                v_id = B[k].pop()
                # ic(v_id)
                count += 1
                pbar.update(1)
                core[k].add(v_id)
                N = set()
                for u_id, layers in MG[v_id].items():
                    for l in layers:
                        Degree[l][u_id] -= 1
                        if I[u_id] > k and Degree[l][u_id] == I[u_id] - 1:
                            N.add(u_id)
                for u_id in N:
                    # ic(u_id)
                    B[I[u_id]].remove(u_id)
                    I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
                    B[I[u_id]].add(u_id)
                MG.remove_node(v_id)
                I[v_id] = 0
                # ic(I)
                # ic(B)
    print_result(core, start_time)


def firm_core2(MG0: nx.MultiGraph, num_layers, lamb):
    MG = MG0.copy()
    start_time = time.time()
    num_nodes = MG.number_of_nodes()
    assert lamb <= num_layers
    core = collections.defaultdict(set)
    _, B, _ = get_IB(MG, num_nodes, 0, lamb)
    count = 0
    with tqdm(range(num_nodes)) as pbar:
        for k in range(num_nodes):
            if count == num_nodes:
                break
            # ic(f'-------------{k}--------------')
            while B[k]:
                # ic(B)
                # ? 需要删除任何I值[不大于k]的顶点, 而不是等于k的顶点
                core[k] = core[k] | B[k]
                pbar.update(len(B[k]))
                # ic(B[k])
                MG.remove_nodes_from(B[k])
                count += len(B[k])
                # ic(MG.number_of_nodes())
                _, B, _ = get_IB(MG, num_nodes, k, lamb)
    print_result(core, start_time)


def firm_core3(MG0: nx.MultiGraph, num_layers, lamb, switch_num=3):
    MG = MG0.copy()
    start_time = time.time()
    num_nodes = MG.number_of_nodes()
    assert lamb <= num_layers
    core = collections.defaultdict(set)
    I, B, _ = get_IB(MG, num_nodes, 0, lamb)
    count = 0
    with tqdm(range(num_nodes)) as pbar:
        for k in range(num_nodes):
            if count == num_nodes:
                break
            # ic(f'-------------{k}--------------')
            phase = True
            while B[k]:
                if phase:
                    # ic(B)
                    core[k] = core[k] | B[k]
                    pbar.update(len(B[k]))
                    # ic(B[k])
                    MG.remove_nodes_from(B[k])
                    count += len(B[k])
                    # ic(MG.number_of_nodes())
                    I, B, Degree = get_IB(MG, num_nodes, k, lamb)
                    if len(B[k]) <= switch_num:
                        phase = False
                else:
                    v_id = B[k].pop()
                    # ic(v_id)
                    count += 1
                    pbar.update(1)
                    core[k].add(v_id)
                    N = set()
                    for u_id, layers in MG[v_id].items():
                        for l in layers:
                            Degree[l][u_id] -= 1
                            if I[u_id] > k and Degree[l][u_id] == I[u_id] - 1:
                                N.add(u_id)
                    for u_id in N:
                        # ic(u_id)
                        B[I[u_id]].remove(u_id)
                        I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
                        B[I[u_id]].add(u_id)
                    MG.remove_node(v_id)
                    I[v_id] = 0
    print_result(core, start_time)


def print_result(core, start_time):
    run_time = time.time() - start_time
    ic(run_time)
    # total_num = 0
    # for k, nodes in core.items():
    #     total_num += len(nodes)
    #     ic(k, len(nodes))
    # ic(total_num)


def get_IB(MG: nx.MultiGraph, num_nodes, k, lamb):
    I = collections.defaultdict(int)

    B = collections.defaultdict(set)
    Degree = utils.get_all_degree(MG, num_nodes, num_layers)
    for v_id in MG.nodes:
        top_d = np.partition(Degree[:, v_id], -lamb)[-lamb]
        assert top_d <= num_nodes
        I[v_id] = top_d
        B[max(top_d, k)].add(v_id)
    return I, B, Degree


# def update_IB(MG: nx.MultiGraph, I, B, Degree, k, v_id, lamb):
#     N = set()
#     for u_id, layers in MG[v_id].items():
#         for l in layers:
#             Degree[l][u_id] -= 1
#             if I[u_id] > k and Degree[l][u_id] == I[u_id] - 1:
#                 N.add(u_id)
#     for u_id in N:
#         # ic(u_id)
#         B[I[u_id]].remove(u_id)
#         I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
#         B[I[u_id]].add(u_id)
#     MG.remove_node(v_id)
#     I[v_id] = 0


if __name__ == '__main__':
    dataset = ['homo', 'sacchcere', 'sanremo', 'slashdot', 'ADHD', 'FAO', 'RM', 'TD']
    # MG = create_data.create_graph()
    MG, num_layers = DS.read_graph(dataset[3], 3)
    ic(num_layers)
    firm_core1(MG, 3, 1)
    firm_core2(MG, 3, 1)
    firm_core3(MG, 3, 1)
    firm_core1(MG, 3, 2)
    firm_core2(MG, 3, 2)
    firm_core3(MG, 3, 2)
    firm_core1(MG, 3, 3)
    firm_core2(MG, 3, 3)
    firm_core3(MG, 3, 3)
