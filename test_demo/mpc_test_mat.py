import numpy as np
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
from nx_graph_test import create_graph_i, get_adj_mat


# 自己预先设置图的节点数和层数
V = 15
L = 3
PADDING = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围

secint4 = mpc.SecInt(16)


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


async def firmcore_mock_adj_mat(lamb):
    # 修改版本, 删去数据结构B, 仅仅使用I
    # 每一方传入的数据是整个图的[邻接矩阵], 之后不再传入其他数据
    # 除了生成v_list, 其他地方尽量不使用明文
    async with mpc:
        l0 = mpc.pid  # 当前层数
        G = create_graph_i(l0)
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        adj_mat = get_adj_mat(G)
        ic(adj_mat)
        adj_mat = adj_mat.flatten().tolist()  # 转化为一维list
        adj_mat = [secint(a) for a in adj_mat]  # [V * V, ]
        adj_mat = mpc.input(adj_mat)  # [L, V * V]
        ic(len(adj_mat), len(adj_mat[0]))
        adj_mat = [seclist(adj_mat[l]) for l in range(L)]  # [L, V * V]
        # ? deg_list列表不使用多方输入, 用每次更新后的整体邻接矩阵计算
        deg_list = [None for _ in range(L)]  # [L, V]
        for l in range(L):
            deg_list[l] = seclist([sum(adj_mat[l][V * i: V * (i + 1)])
                                   for i in range(V)])
        ic(deg_list)
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I')
        for i in range(V):
            # 每个id在各层的度列表
            per_id = [deg_list[l][i] for l in range(L)]
            I[i] = mpc.sorted(per_id)[-lamb]  # id为i的顶点在各层的第λ大的度
        ic(await mpc.output(list(I)))
        core = [[secint(PADDING)] for _ in range(V - 1)]  # 最终要返回的加密的结果
        # 还未被移除的v
        remain_v = seclist([i for i in range(V)], secint4)
        for k in range(V - 1):
            ic(f'---------{k}---------')
            ic('获取I中值等于k的顶点')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, -1)
                              for v_id in range(V)], secint4)
            v_list.sort()
            n = await mpc.output(v_list.count(-1))
            ic(n)
            del v_list[:n]
            ic(v_list)
            ic('遍历所有I值等于k的顶点')
            # 后面可能还会有新的顶点加入v_list
            while v_list:
                # 在每一层分别中去掉这个节点
                ic('移除选中的顶点')
                v_id = v_list.pop(0)
                remain_v.remove(v_id)
                core[k].append(v_id)
                # 在邻接矩阵中移除节点v
                for l in range(L):
                    for i in range(V):
                        adj_mat[l][v_id * V + i] = secint(0)
                deg_list = [None for _ in range(L)]  # [L, V]
                for l in range(L):
                    deg_list[l] = seclist([sum(adj_mat[l][V * i: V * (i + 1)])
                                           for i in range(V)])
                I = seclist([-1 for _ in range(V)], secint4)
                ic('更新数组I')
                # 只需要重新计算还存在的节点
                for v_id in remain_v:  # 每个id在各层的度列表
                    per_id = [deg_list[l][i] for l in range(L)]
                    I[i] = mpc.sorted(per_id)[-lamb]  # id为i的顶点在各层的第λ大的度
        core = [await mpc.output(c) for c in core]
        ic(core)


# async def firmcore_mock_adj_mat(lamb):
#     # 修改版本, 删去数据结构B, 仅仅使用I
#     # 每一方传入的数据是整个图的[邻接矩阵], 之后不再传入其他数据
#     # ? 这里尝试将邻接矩阵全部展平
#     async with mpc:
#         l = mpc.pid  # 当前层数
#         G = create_graph_i(l)
#         # 每个顶点vertices对应的Top-λ(deg(vertices))
#         ic(f'{mpc.pid + 1} of {L} party')
#         # 获取整个多层图的度矩阵
#         adj_mat = get_adj_mat(G)
#         deg_list = get_degree(G)
#         ic(deg_list)
#         ic(adj_mat)
#         deg_list = [secint(deg_list[i]) for i in range(V)]  # V
#         adj_mat = adj_mat.flatten().tolist()  # 转化为一维list
#         adj_mat = [secint(a) for a in adj_mat]  # V * V
#         adj_mat = mpc.input(adj_mat)  # (L, V * V)
#         ic(len(adj_mat), len(adj_mat[0]))
#         adj_mat_flat = seclist(sectype=secint4)  # 将邻接矩阵展平成一维矩阵 L * V * V
#         for a in adj_mat:
#             adj_mat_flat += a
#         ic(len(adj_mat_flat))
#         deg_list = mpc.input(deg_list)  # (L, V)
#         deg_list_flat = seclist(sectype=secint4)  # L * V
#         for a in deg_list:
#             deg_list_flat += a
#         ic(len(deg_list_flat))
#         I = seclist([-1 for _ in range(V)], secint4)
#         ic('初始化数组I')
#         for i in range(V):  # 每个id在各层的度列表
#             per_id = [deg_list_flat[l * V + i] for l in range(L)]
#             I[i] = mpc.sorted(per_id)[-lamb]  # id为i的顶点在各层的第λ大的度
#         ic(await mpc.output(list(I)))
#         core = [[secint(PADDING)] for _ in range(V - 1)]  # 最终要返回的加密的结果
#         for k in range(V - 1):
#             ic(f'---------{k}---------')
#             ic('获取I中值等于k的顶点')
#             v_list = seclist([mpc.if_else(I[v_id] == k, v_id, -1)
#                               for v_id in range(V)], secint4)
#             v_list.sort()
#             n = await mpc.output(v_list.count(-1))
#             ic(n)
#             del v_list[:n]
#             ic(v_list)
#             ic('遍历所有I值等于k的顶点')
#             # 后面可能还会有新的顶点加入v_list
#             while v_list:
#                 # 在每一层分别中去掉这个节点
#                 ic('移除选中的顶点')
#                 v_id = v_list.pop(0)
#                 core[k].append(v_id)
#                 # 在邻接矩阵中移除节点v
#                 for i in range(V):
#                     adj_mat_flat[l * V * V + v_id * V + i] = secint(0)
#                 # deg_list =
#                 ic('获取v_id的邻居, 度--')
#                 for l in range(L):
#                     # 第l层的邻接矩阵
#                     for u_id in range(V):
#                         # 第l层节点u_id和节点v_id是否是邻居
#                         deg_list_flat[l * V + u_id] -= mpc.if_else(
#                             adj_mat_flat[l * V * V + v_id * V + u_id] == 1, 1, 0)
#                         updated.append(mpc.if_else(
#                             (I[u_id] > k) & (I[u_id] - 1 == deg_list_flat[l * V + u_id]), u_id, PADDING))
#                 for u_id in updated:
#                     per_id = [deg_list_flat[l * V + u_id] for l in range(L)]
#                     # 即使u_id为-1, 该等式也能执行, 只是多了一次无用计算
#                     I[u_id] = mpc.if_else(
#                         u_id != PADDING, mpc.sorted(per_id)[-lamb], I[u_id])
#         core = [await mpc.output(c) for c in core]
#         ic(core)

if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mock_adj_mat(2))
