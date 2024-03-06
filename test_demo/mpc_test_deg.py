# 暂时不使用图数据库, 而使用单独的图结构作为各方的数据源

import numpy as np
import networkx as nx
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
from nx_graph_test import create_graph_i, get_degree

# 自己预先设置图的节点数和层数
V = 15
L = 3
PADDING = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围

secint4 = mpc.SecInt(16)


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


async def firmcore_mock_degree2_0(lamb):
    # ? 修改版本, 删去数据结构B, 仅仅使用I
    # I: 每个顶点vertices对应的Top-λ(deg(vertices))
    async with mpc:
        I = [None for _ in range(V)]
        G = create_graph_i(mpc.pid)
        assert (isinstance(G, nx.MultiDiGraph))
        # 该层所有的度, list[V * int]
        deg_list = get_degree(G)
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        deg_list = get_degree(G)
        assert isinstance(deg_list, list)
        # 换成安全类型的list
        deg_list = [secint(d) for d in deg_list]
        Degree = np.array(mpc.input(deg_list))
        # Degree的[行]为层数, [列]为id
        ic(len(Degree))
        ic(len(Degree[0]))
        Degree = [seclist(per_layer, secint4) for per_layer in Degree]
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度
            per_id = [Degree[l][i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        # output时必须是list形式
        ic(await mpc.output(list(I)))
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(PADDING)] for _ in range(V - 1)]
        for k in range(V - 1):
            # ic(f'---------{k}---------')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, PADDING)
                             for v_id in range(V)], secint4)
            n = await mpc.output(v_list.count(PADDING))
            # ic(n)
            # 得到的v_list前l位不为0, 后面全为0, 截取前面不为0的位
            v_list.sort()
            del v_list[:n]
            v_list_out = await mpc.output(list(v_list))
            ic(v_list_out)
            # ? 难点: 每次删除顶点, 更新度之后v_list也需要更新
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回v的邻居节点id, 再次input
                v_id0 = await mpc.output(v_id)
                G.remove_node(v_id0)
                # 被移除节点的邻居节点集合
                # ? 直接把所有的度再重新传一遍
                deg_list = get_degree(G)
                deg_list = [secint(d) for d in deg_list]
                Degree = np.array(mpc.input(deg_list))
                # Degree的[行]为层数, [列]为id
                Degree = [seclist(per_layer, secint4) for per_layer in Degree]
                # 每个顶点vertices对应的Top-λ(deg(vertices))
                # ic('更新数组I')
                I = seclist([-1 for _ in range(V)], secint4)
                for i in range(V):
                    # 每个id在各层的度
                    per_id = [Degree[l][i] for l in range(L)]
                    # id为i的顶点在各层的第λ大的度
                    I[i] = mpc.sorted(per_id)[-lamb]
                for i in G.nodes:
                    v_list.append(mpc.if_else((I[i] == k) & (
                        v_list.count(i) == 0), i, PADDING))
                n = await mpc.output(v_list.count(PADDING))
                v_list.sort()
                del v_list[:n]
                v_list_out = await mpc.output(list(v_list))
                ic(v_list_out)
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firmcore_mock_degree2_2(lamb):
    # 上面函数的mock测试版本
    # 修改版本, 删去数据结构B, 仅仅使用I
    # ? 前面的-1都保留下来, 等到最后再处理, 除了选择v_list时, 其他地方避免使用明文
    # 修改Degree矩阵结构为[seclist, seclist, ...], 每个seclist为每个id在各层的节点
    G = create_graph_i()
    context = mgp.ProcCtx(G)

    async with mpc:
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        deg_list = firmcore_mage.get_all_deg(
            context, mpc.pid).fields['deg_list']
        assert isinstance(deg_list, list)
        # 换成安全类型的list
        secure_deg_list = [secint(d)
                           for d in deg_list] + [secint(PADDING)]
        # * 从mpc.input输出的数据都是加密的了, 因为获取了其他方的数据
        # Degree指的是所有的度, 形状为 (V + 1) * L, 最后一个数字用于填充
        Degree = np.array(mpc.input(secure_deg_list))
        # Degree的[行]为层数, [列]为id
        Degree = [seclist(per_layer, secint4) for per_layer in Degree]
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度
            per_id = [Degree[l][i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        # output时必须是list形式
        ic(await mpc.output(list(I)))
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(PADDING)] for _ in range(V - 1)]
        ic('下面是根据k逐渐增大来依次淘汰顶点')
        for k in range(V - 1):
            ic(f'---------{k}---------')
            v_list = []
            for v_id in range(V):
                if await mpc.eq_public(I[v_id], k):
                    v_list.append(v_id)
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                # ic(await mpc.output(v_id))
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回v的邻居节点id, 再次input
                # v_id0 = await mpc.output(v_id)
                ic('删除这次选中的节点, 并修改其邻居的度')
                updated = firmcore_mage.remove_node(
                    context, mpc.pid, v_id).fields['updated']
                ic(updated)
                assert isinstance(updated, list)
                # 所有需要修改degree和I[v_id]的节点id
                secure_updated = []
                for u_id in updated:
                    if await mpc.eq_public((I[u_id] > k), True):
                        secure_updated.append(secint(u_id))
                # 输入的数组不为空
                if not secure_updated:
                    secure_updated.append(secint(PADDING))
                secure_updated = mpc.input(secure_updated)
                u_all_layer = seclist([], secint4)
                ic('遍历每层修改过度的节点')
                for l, u_per_layer in enumerate(secure_updated):
                    # 每一个需要修改I的u_id
                    for u_id in u_per_layer:
                        # 首先需要先更新Degree矩阵
                        # u_id0 = await mpc.output(u_id)
                        if not mpc.eq_public(u_id, -1):
                            Degree[l][u_id] -= 1
                            # * 不同的层之间传的节点可能有重复
                            if await mpc.eq_public(Degree[l][u_id], I[u_id] - 1):
                                u_all_layer.append(u_id)
                ic(u_all_layer)
                if u_all_layer:
                    for u_id in u_all_layer:
                        per_id = [Degree[l][u_id] for l in range(L)]
                        # id为i的顶点在各层的第λ大的度
                        I[u_id] = mpc.sorted(per_id)[-lamb]
        core = [await mpc.output(c) for c in core]
        ic(core)

if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mock_degree2_0(2))
