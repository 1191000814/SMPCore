"""
论文firmcore decomposition of multilayer networks的实现
(有向图部分)
"""

import mem_demo.dataset as dataset
import utils
import numpy as np
from icecream import ic
import networkx as nx
import time
import collections


def firm_d_core(MDG: nx.MultiDiGraph, L, lamb):
    '''
    返回多层有向图中(lamb, k)-FirmD-Core的顶点集合
    '''
    t_begin = time.time()
    V = MDG.number_of_nodes()
    assert lamb <= L
    # 每个顶点v对应的Top_d-和Top_d+
    I_in = [-1 for _ in range(V)]

    # B只用记录T中节点的in度
    B_in = collections.defaultdict(set)

    # V * V的字典
    # k, r存储在V*k+r中
    core = collections.defaultdict(set)

    # 枚举第一个参数k(0..V-1)
    for k in range(V):
        ic(f'-------{k}------')
        # S: 源节点, 初始化为 所有出度大于k的节点id
        # T: 目标节点, 初始化为 全部节点
        # 用v和u表示节点, 方向为 v(S)->u(T)
        T = set([i for i in range(V)])
        # H: S和T的导出子图, S: 满足条件的源节点
        H, S, I_out = utils.get_subgraph(MDG, k, lamb, V, L)
        # 度矩阵, Degree[l][id]为第l层标识符为id的顶点的(0: in度, 1: out度)
        # 每轮迭代都会删除图中顶点, 所以需要重新计算
        Degree = utils.get_all_degree(H, V, L)
        for v in T:  # 目标顶点集/入边
            I_in[v] = np.partition(Degree[:, v, 0], -lamb)[-lamb]
            B_in[I_in[v]].add(v)
        # 枚举第二个参数r(0..V-1)
        for r in range(V):
            # ic(f'---{k}, {r}---')
            while B_in[r]:
                # v是T中的元素
                # ic(B)
                # ic(T)
                v = B_in[r].pop()
                # ic(v)
                # ? 移除T中元素v时, 同时需要移除所有v的in边
                # ? 但是不能在H中移除节点v, 因为它可能还存在于集合S中
                T.remove(v)
                core[k * V + r].add(v)
                N = set()
                # 修改v所有[前驱节点]u(都在S中)的度, 并记录哪些u的I+需要修改
                for u in H.predecessors(v):
                    assert u in S
                    for l in H[u][v]:  # u->v在哪些层存在
                        if I_out[u] > k:
                            Degree[l][u][1] -= 1
                            if Degree[l][u][1] == I_out[u] - 1:
                                N.add(u)
                if v in S:
                    H.remove_edges_from([(u, v) for u, v in H.in_edges(v)])
                else:
                    H.remove_node(v)
                # S中需要修改I_out[u_id]值的节点
                for u in N:
                    I_out[u] = np.partition(Degree[:, u, 1], -lamb)[-lamb]
                    # 如果更新之后的I值小于k, 在S中删除该节点, 并更新其所有后继节点的度
                    if I_out[u] < k:
                        S.remove(u)
                        for v1 in H[u]:
                            for l in H[u][v1]:  # u->v1在哪些层存在
                                if I_in[v1] > r:
                                    Degree[l][v1][0] -= 1
                                    if Degree[l][v1][0] == I_in[v1] - 1:
                                        B_in[I_in[v1]].remove(v1)
                                        I_in[v1] = np.partition(Degree[:, v1, 0], -lamb)[-lamb]
                                        B_in[I_in[v1]].add(v1)
                        # ? 移除S中元素u时, 同时要移除u中所有的out边
                        if u in T:
                            H.remove_edges_from([(u, v)
                                                for u, v in H.out_edges(u)])
                        else:
                            H.remove_node(u)
    t_end = time.time()
    ic(t_end - t_begin)
    return core


if __name__ == '__main__':
    MDG, L = dataset.create_by_file('homo', di=True)
    MDG = dataset.create_graph(di=True)
    # V = MDG.number_of_nodes()
    core_d = firm_d_core(MDG, L, 2)
    ic(core_d)
