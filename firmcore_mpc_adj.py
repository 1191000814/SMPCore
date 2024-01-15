# 每一方都在自己本地运行该代码, 只能从input中获取其他方的数据(加密)
# 只使用自己数据计算时, 无需加密
# 将数据传给其他方, 或者获取其他方的数据时, 都需要加密

import numpy as np
import mgp_mock as mgp
import firmcore_mage
from neo4j import GraphDatabase
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
from nx_graph import create_3layer_1

URI = "bolt://localhost:7687"
AUTH = ('', '')

# 自己预先设置图的节点数和层数
V = 15
L = 3
PADDING = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围

secint4 = mpc.SecInt(16)


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


async def firmcore_mock_adj(lamb):
    # 修改版本, 删去数据结构B, 仅仅使用I
    # 每一方传入的数据是整个图的邻接表, 之后不再传入其他数据
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
        input_list = adj_list
        # 前V项是adj表, 第V+1项是deg表
        input_list.append(deg_list)
        input_list = [mpc.input(a) for a in input_list]
        # deg_list: list[seclist, seclist...], deg_list[l][v_id]为第l层v_id顶点的度
        deg_list = [seclist(per_layer, secint4)
                    for per_layer in input_list[-1]]
        # adj_list: [seclist[list[], ...], ...], adj_list[l][v_id]为第l层v_id顶点的邻居列表
        adj_list = [seclist(per_layer, list) for per_layer in input_list[:-1]]
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
        ic(await mpc.output(I))
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(PADDING)] for _ in range(V - 1)]
        # 下面是根据k逐渐增大来依次淘汰顶点
        for k in range(V - 1):
            ic(f'---------{k}---------')
            v_list = seclist([mpc.if_else(top_d == k, v_id, PADDING)
                             for v_id, top_d in enumerate(I)], secint4)
            n = await mpc.output(v_list.count(PADDING))
            # 得到的v_list前l位不为0, 后面全为0, 截取前面不为0的位
            v_list.sort()
            del v_list[:n]
            v_list_out = await mpc.output(list(v_list))
            ic(v_list_out)
            v_list = list(v_list)
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                core[k].append(v_id)
                # 所有需要修改I[v_id]的节点id
                updated = []
                for l, per_layer in enumerate(adj_list):
                    ic(type(per_layer))
                    neighbors = per_layer[v_id]
                    for u_id in neighbors:
                        deg_list[l][u_id] -= mpc.if_else(I[u_id] > k, 1, 0)
                        updated.append(mpc.if_else(
                            I[u_id] > k and I[u_id] - 1 == deg_list[l][u_id], u_id, PADDING))
                for u_id in updated:
                    per_id = [deg_list[l][i] for l in range(L)]
                    # 即使u_id为-1, 该等式也能执行, 只是多了一次无用计算
                    I[u_id] = mpc.if_else(
                        u_id != PADDING, mpc.sorted(per_id)[-lamb], I[u_id])
        core = [await mpc.output(c) for c in core]
        ic(core)


if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mock_adj(2))
