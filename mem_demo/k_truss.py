import mem_demo.dataset as dataset
import utils
import numpy as np
from icecream import ic
import networkx as nx
import os
import time
import collections


class Edge():
    '''
    无向边
    '''

    def __init__(self, u, v):
        self.u = min(u, v)
        self.v = max(u, v)

    def __eq__(self, __value: object) -> bool:
        return self.u == __value.u and self.v == __value.v

    def __hash__(self) -> int:
        return hash((self.u, self.v))


def truss_single(MG: nx.MultiGraph, l):
    '''
    计算多层图中某一层的k_truss
    每条边e的最大k值, T[e]表示最大能包含e的k值
    '''
    G = nx.Graph()  # 获取该多层图的第l层, 深copy一遍, 防止改变多层图的结构
    for u, v, layer in MG.edges:
        if layer == l:
            # 计算每条边被包含的三角形个数
            G.add_edge(u, v)
    T_e = collections.defaultdict(0)  # 边Edge(u, v) -> T_l(e)
    # 边所在的三角形个数 -> 边(节点id对)集合
    num_tri = get_num_tri(G)
    while num_tri:
        k = min(num_tri.keys())
        for e in num_tri[k]:
            G.remove_edge(*e)
            T_e[Edge(*e)] = k + 2
        num_tri.pop(k)
        num_tri = get_num_tri(G)
    T_v = {}
    for v in G.nodes:
        T_v[v] = max([T_e[Edge(*e)] for e in G[v]])
    return T_v


def truss_multi():
    '''
    多层图上的truss
    '''


def get_num_tri(G):
    num_tri = collections.defaultdict(set)
    for u, v in G.edges:
        # 计算每条边被包含的三角形个数
        k = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        num_tri[k].add((u, v))
    return num_tri


if __name__ == '__main__':
    e1 = Edge(1, 2)
    e2 = Edge(2, 1)
    d = {e1: 2}
    ic(d[e2])
