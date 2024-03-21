# 暂时不使用图数据库, 而使用单独的图结构作为各方的数据源np
# 无向图-smpc-firmcore算法

import numpy as np
import networkx as nx
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
import create_data
import utils
import collections

# 自己预先设置图的节点数和层数
# V = 15
# L = 3
pad_val = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围

secint4 = mpc.SecInt(16)


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


async def firm_core(lamb):
    # ? delete B, only use I
    # I: 每个顶点vertices对应的Top-λ(deg(vertices))
    async with mpc:
        G = create_data.create_layer(mpc.pid)
        assert (isinstance(G, nx.Graph))
        num_layers = len(mpc.parties)  # 层数
        num_nodes = G.number_of_nodes()  # 节点数
        ic(f'{mpc.pid + 1} of {num_layers} party')
        deg_list = utils.get_degree(G, num_nodes)  # 该层的节点的度列表
        deg_list = [secint(deg) for deg in deg_list]  # 转化成安全矩阵
        Degree = np.array(mpc.input(deg_list))  # 整个图的度矩阵
        # Degree的[行]为层数, [列]为id
        ic(len(Degree))
        ic(len(Degree[0]))
        Degree = [seclist(per_layer, secint4) for per_layer in Degree]
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic('创建数组I')
        I = seclist([pad_val for _ in range(num_nodes)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(num_nodes):
            # 每个id在各层的度
            per_id = [Degree[l][i] for l in range(num_layers)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        # output时必须是list形式
        ic(await mpc.output(list(I)))
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(pad_val)] for _ in range(num_nodes - 1)]
        k = 0
        counted = 0  # 已经计算了的顶点
        while (k < num_nodes) and (counted < num_nodes):
            ic(f'---------{k}---------')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, pad_val)
                             for v_id in range(num_nodes)], secint4)
            n = await mpc.output(v_list.count(pad_val))
            # ic(n)
            # 得到的v_list前l位不为0, 后面全为0, 截取前面不为0的位
            v_list.sort()
            del v_list[:n]
            # v_list_out = await mpc.output(list(v_list))
            # ic(v_list_out)
            # ? 难点: 每次删除顶点, 更新度之后v_list也需要更新
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                # ? 该节点的真实值在最后也将暴露给用户, 所以可以在该步解密
                core[k].append(v_id)
                counted += 1
                # 每一方更新自己层的顶点的度, 返回v的邻居节点id, 再次input
                v_id0 = await mpc.output(v_id)
                G.remove_node(v_id0)
                # 被移除节点的邻居节点集合
                # ? 直接把所有的度再重新传一遍
                deg_list = utils.get_degree(G, num_nodes)
                deg_list = [secint(d) for d in deg_list]
                Degree = np.array(mpc.input(deg_list))
                # ic(Degree.shape)
                # Degree的[行]为层数, [列]为id
                Degree = [seclist(per_layer, secint4) for per_layer in Degree]
                # 每个顶点vertices对应的Top-λ(deg(vertices))
                # ic('更新数组I')
                I = seclist([pad_val for _ in range(num_nodes)], secint4)
                for i in range(num_nodes):
                    # 每个id在各层的度
                    per_id = [Degree[l][i] for l in range(num_layers)]
                    # id为i的顶点在各层的第λ大的度
                    # 这里不需要考虑I[i]>k, 因为v_list只多不少
                    I[i] = mpc.sorted(per_id)[-lamb]
                for i in G.nodes:
                    v_list.append(mpc.if_else((I[i] == k) & (
                        v_list.count(i) == 0), i, pad_val))
                if v_list:
                    n = await mpc.output(v_list.count(pad_val))
                    v_list.sort()
                    del v_list[:n]
                    # v_list_out = await mpc.output(list(v_list))
                    # ic(v_list_out)
                else:
                    break
            k += 1
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firm_core_mod(lamb, dataset=None):
    # ? 仅保护Degree的度不被泄露, 中间结果I可以被各方知道
    async with mpc:
        l0 = mpc.pid  # 该层id
        if dataset is None:
            G = create_data.create_layer(l0)  # 该计算方的单层图
        else:
            G = create_data.create_layer_by_file(dataset, mpc.pid)
        num_layers = len(mpc.parties)  # 层数
        num_nodes = G.number_of_nodes()  # 节点数
        l0 = mpc.pid  # 该层id
        ic(f'{l0 + 1} of {num_layers} party')
        deg_list = utils.get_degree(G, num_nodes)  # 该层的节点的度列表
        remain_v = set([i for i in range(num_nodes)])  # 还没被移除的点
        ic('init i, B')
        I, B = await get_IB(deg_list, lamb, remain_v)
        core = collections.defaultdict(set)
        # 从0开始依次移除最小Top-d的顶点
        for k in range(num_nodes):
            ic(f'--------{k}------')
            while B[k]:
                # ic(I)
                # ic(B)
                v_id = B[k].pop()
                # ic(v_id)
                ic(len(remain_v))
                core[k].add(v_id)
                remain_v.remove(v_id)
                # 移除v后, 将需要修改top_d的顶点存储在N中
                update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                for u_id in set(G.neighbors(v_id)) & remain_v:
                    deg_list[u_id] -= 1
                    if deg_list[u_id] == I[u_id] - 1 and I[u_id] > k:
                        update_v.add(u_id)
                # ic(update_v)
                await update_IB(deg_list, lamb, list(update_v), I, B)
        ic(core)


async def firm_core_mod1(lamb, dataset=None):
    # ? 每次删除全部B[k]节点
    async with mpc:
        l0 = mpc.pid  # 该层id
        if dataset is None:
            G = create_data.create_layer(l0)  # 该计算方的单层图
        else:
            G = create_data.create_layer_by_file(dataset, mpc.pid)
        num_layers = len(mpc.parties)  # 层数
        num_nodes = G.number_of_nodes()  # 节点数
        l0 = mpc.pid  # 该层id
        ic(f'{l0 + 1} of {num_layers} party')
        deg_list = utils.get_degree(G, num_nodes)  # 该层的节点的度列表
        remain_v = set([i for i in range(num_nodes)])  # 还没被移除的点
        ic('init i, B')
        I, B = await get_IB(deg_list, lamb, remain_v)
        core = collections.defaultdict(set)
        # 从0开始依次移除最小Top-d的顶点
        for k in range(num_nodes):
            ic(f'--------{k}------')
            while B[k]:
                # ic(I)
                # ic(B)
                v_id = B[k].pop()
                # ic(v_id)
                ic(len(remain_v))
                core[k].add(v_id)
                remain_v.remove(v_id)
                # 移除v后, 将需要修改top_d的顶点存储在N中
                update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                for u_id in set(G.neighbors(v_id)) & remain_v:
                    deg_list[u_id] -= 1
                    if deg_list[u_id] == I[u_id] - 1 and I[u_id] > k:
                        update_v.add(u_id)
                # ic(update_v)
                await update_IB(deg_list, lamb, list(update_v), I, B)
        ic(core)


async def get_IB(deg_list: list, lamb, remain_v):
    '''
    初始化I和B
    根据当前层的度矩阵, 获取整个多层图的B结构
    '''
    num_nodes = len(deg_list)
    ic('get local degree list')
    deg_list = [secint(deg) for deg in deg_list]  # 转化成安全矩阵
    # Degree的[行]为层数, [列]为id
    ic('get all degree list')
    Degree = np.array(mpc.input(deg_list))  # 整个图的度矩阵
    # 每个顶点vertices对应的Top-λ(deg(vertices))
    num_layers = len(Degree)
    I = [pad_val for _ in range(num_nodes)]
    ic('compute I')
    for i in range(num_nodes):
        # 每个id在各层的度
        per_id = [Degree[l][i] for l in range(num_layers)]
        # id为i的顶点在各层的第λ大的度
        I[i] = mpc.sorted(per_id)[-lamb]
    # ? 明文计算I中的内容
    ic('get plaintext I')
    I = await mpc.output(I)
    # ? 既然I是明文的, 那么也可以有B
    B = collections.defaultdict(set)
    # 只计算剩余节点, 添加进B
    ic('compute B')
    for v_id in remain_v:
        # id为i的顶点在各层的第λ大的度
        B[I[v_id]].add(v_id)
    return I, B


async def update_IB(deg_list: list, lamb, update_v: list, I, B):
    '''
    根据当前层的度矩阵, 获取整个多层图的B结构
    '''
    # ? 在mod1的基础上进行修改, 不重新计算所有的I的值, 只计算必须修改I值节点的I值
    ic('get local degree list')
    deg_list = [secint(deg) for deg in deg_list]  # 转化成安全矩阵
    # Degree的[行]为层数, [列]为id
    ic('get all degree list')
    Degree = np.array(mpc.input(deg_list))  # 整个图的度矩阵
    # 每个顶点vertices对应的Top-λ(deg(vertices))
    num_layers = len(Degree)
    ic('collect all v need to be updated')
    # ic(update_v)
    # 这里需要使得每个mpc.input的数量都相等
    update_nums = mpc.input(secint(len(update_v)))  # 单层图需要修改I的数量
    max_update_num = await mpc.output(mpc.max(update_nums))  # 各层中最大需要修改的数量
    if (max_update_num == 0):  # 传入的长度至少为1
        max_update_num = 1
    update_v.extend([pad_val] * (max_update_num - len(update_v)))  # 填充到最大数量
    all_update_v = mpc.input([secint(v_id) for v_id in update_v])  # 各层需要改变I的节点
    # ic(len(all_update_v))
    # ic(all_update_v)
    all_update_v = [v_id for per_layer in all_update_v for v_id in per_layer]  # 全放到一个列表里
    # ic(len(all_update_v))
    # ic(all_update_v)
    all_update_v = set(await mpc.output(all_update_v))  # 明文后用集合表示
    #! pad_val有可能不存在
    all_update_v.discard(pad_val)
    ic(all_update_v)
    ic('update I and B')
    # ? 明文计算I中的内容
    for v_id in all_update_v:
        # 每个id在各层的度
        per_id = [Degree[l][v_id] for l in range(num_layers)]
        # id为i的顶点在各层的第λ大的度
        B[I[v_id]].remove(v_id)
        I[v_id] = await mpc.output(mpc.sorted(per_id)[-lamb])
        B[I[v_id]].add(v_id)

if __name__ == '__main__':
    mpc.run(firm_core_mod1(2, 'homo'))
