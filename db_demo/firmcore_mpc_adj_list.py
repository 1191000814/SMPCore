import numpy as np
import mgp_mock as mgp
import firmcore_mage
from sys import exit
from neo4j import GraphDatabase
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from mpyc.sectypes import SecureInteger
from icecream import ic
from test_demo.nx_graph_test import create_3layer_1

# 自己预先设置图的节点数和层数
V = 15
L = 3
PADDING = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围

secint4 = mpc.SecInt(16)


class SecureNode:
    # 用来在邻接表中存储邻居的链表节点
    def __init__(self, value):
        # ic(value)
        if isinstance(value, int):
            value = secint(value)
        self.value = value
        self.next = None

    def __add__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value + other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value + other.value)
        else:
            raise TypeError('unsupport type for add')

    def __sub__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value - other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value - other.value)
        else:
            raise TypeError('unsupport type for sub')

    def __mul__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value * other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value * other.value)
        else:
            raise TypeError('unsupport type for mul')

    def __rmul__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value * other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value * other.value)
        else:
            raise TypeError('unsupport type for rmul')


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


def get(ls, ix):
    # 使用加密索引ix获取ls中的复杂对象
    # 当数组ls中元素不为整数或者安全类型时, seclist是不支持根据[加密索引]获取对象的
    assert isinstance(ls, list) or isinstance(ls, seclist)
    if isinstance(ix, int):
        return ls[ix]
    l = len(ls)
    if l == 0:
        raise ValueError('the list is empty')
    assert isinstance(ls[0], SecureNode)
    res = SecureNode(-1)
    for i, node in enumerate(ls):
        res = mpc.if_else(i == ix, node, res)
    return res


async def firmcore_mock_adj_list(lamb):
    # 修改版本, 删去数据结构B, 仅仅使用I
    # 每一方传入的数据是整个图的[邻接表], 之后不再传入其他数据
    # 除了生成v_list, 其他地方尽量不使用明文
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        records = firmcore_mage.get_graph_adj(context, mpc.pid)
        # list[int], 每个元素是一个顶点的度
        deg_list, adj_list = records.fields['deg_list'], records.fields['adj_list']
        ic(deg_list, adj_list)
        deg_list = [secint(deg) for deg in deg_list]  # 度列表
        # list[list[int], ...], 每个元素是其邻居列表, 空列表则填充-1
        for v_id in range(V):
            adj_list[v_id] = [secint(u_id) for u_id in (
                adj_list[v_id] if adj_list[v_id] else [-1])]
        deg_list = mpc.input(deg_list)
        adj_list = [mpc.input(a) for a in adj_list]
        ic(len(adj_list), len(adj_list[0]))
        # deg_list: list[seclist, seclist...], deg_list[l][v_id]为第l层v_id顶点的度
        deg_list = [seclist(per_layer, secint4)
                    for per_layer in deg_list]
        # adj_list[l][v_id]表示为第l层v_id节点的所有邻居, 其中邻居用链表节点SecureNode表示
        adj_list_temp = [seclist([SecureNode(None) for _ in range(V)], SecureNode)
                         for _ in range(L)]
        ic(len(adj_list), len(adj_list[0]))
        for v_id, per_id in enumerate(adj_list):
            for l, neighbors in enumerate(per_id):
                node = adj_list_temp[l][v_id]
                assert isinstance(node, SecureNode)
                # 第l层第v_id个顶点的邻居
                for u_id in neighbors:
                    node.num = u_id
                    node.next = SecureNode(None)
        adj_list = adj_list_temp
        # ic(deg_list)
        # ic(adj_list)
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度列表
            per_id = [deg_list[l][i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        ic(await mpc.output(list(I)))
        # core是最终要返回的加密的结果
        core = [[secint(PADDING)] for _ in range(V - 1)]
        ic('根据k逐渐增大来依次淘汰顶点')
        for k in range(V - 1):
            ic(f'---------{k}---------')
            ic('获取与I中值等于k的顶点')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, -1)
                              for v_id in range(V)], secint4)
            v_list_out = await mpc.output(list(v_list))
            ic(v_list_out)
            v_list.sort()
            n = await mpc.output(v_list.count(-1))
            ic(n)
            del v_list[:n]
            ic(v_list)
            ic('遍历所有I值等于k的顶点')
            while v_list:
                # 在每一层分别中去掉这个节点
                ic('移除选中的顶点')
                v_id = v_list.pop(0)
                core[k].append(v_id)
                # 所有需要修改I[v_id]的节点id
                updated = []
                ic('获取该移除顶点的邻居, 度--')
                for l, per_layer in enumerate(adj_list):  # 每一层的顶点
                    node = get(per_layer, v_id)  # 每一层中的顶点v_id
                    while node is not None and node.value is not None:
                        assert isinstance(node, SecureNode)
                        u_id = node.num
                        deg_list[l][u_id] -= 1
                        updated.append(mpc.if_else(
                            (I[u_id] > k) & (I[u_id] - 1 == deg_list[l][u_id]), u_id, PADDING))
                        node = node.next
                ic('更新某些邻居节点的I值')
                for u_id in updated:
                    per_id = [deg_list[l][i] for l in range(L)]
                    # 即使u_id为-1, 该等式也能执行, 只是多了一次无用计算
                    I[u_id] = mpc.if_else(
                        u_id != PADDING, mpc.sorted(per_id)[-lamb], I[u_id])
        core = [await mpc.output(c) for c in core]
        ic(core)


if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mock_adj_list(2))
