import numpy as np
import mgp_mock as mgp
import firmcore_mage
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
from test_demo.nx_graph_test import create_3layer_1


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
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        records = firmcore_mage.get_adj_mat(context, mpc.pid)
        # list[int], 每个元素是一个顶点的度
        adj_mat = records.fields['adj_mat']
        assert isinstance(adj_mat, np.ndarray)
        ic(adj_mat)
        deg_list = [secint(int(np.sum(adj_mat[:, i]))) for i in range(V)]  # V
        adj_mat = adj_mat.flatten().tolist()  # 转化为一维list
        adj_mat = [secint(a) for a in adj_mat]  # V * V
        adj_mat = mpc.input(adj_mat)  # (L, V * V)
        ic(len(adj_mat), len(adj_mat[0]))
        adj_mat_flat = seclist(sectype=secint4)  # 将邻接矩阵展平成一维矩阵 L * V * V
        for a in adj_mat:
            adj_mat_flat += a
        ic(len(adj_mat_flat))
        deg_list = mpc.input(deg_list)  # (L, V)
        deg_list_flat = seclist(sectype=secint4)  # L * V
        for a in deg_list:
            deg_list_flat += a
        ic(len(deg_list_flat))
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度列表
            per_id = [deg_list_flat[l * V + i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        ic(await mpc.output(list(I)))
        # core是最终要返回的加密的结果
        core = [[secint(PADDING)] for _ in range(V - 1)]
        ic('根据k逐渐增大来依次淘汰顶点')
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
                core[k].append(v_id)
                # 所有需要修改I[v_id]的节点id
                updated = []
                ic('获取v_id的邻居, 度--')
                for l in range(L):
                    # 第l层的邻接矩阵
                    for u_id in range(V):
                        # 第l层节点u_id和节点v_id是否是邻居
                        deg_list_flat[l * V + u_id] -= mpc.if_else(
                            adj_mat_flat[l * V * V + v_id * V + u_id] == 1, 1, 0)
                        updated.append(mpc.if_else(
                            (I[u_id] > k) & (I[u_id] - 1 == deg_list_flat[l * V + u_id]), u_id, PADDING))
                ic('更新某些邻居节点的I值')
                for u_id in updated:
                    per_id = [deg_list_flat[l * V + u_id] for l in range(L)]
                    # 即使u_id为-1, 该等式也能执行, 只是多了一次无用计算
                    I[u_id] = mpc.if_else(
                        u_id != PADDING, mpc.sorted(per_id)[-lamb], I[u_id])
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firmcore_mock_adj_mat2(lamb):
    # 每一方传入的数据是整个图的[邻接矩阵], 之后不再传入其他数据
    # * 任何地方都不使用明文, 时间复杂度可能增大
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        records = firmcore_mage.get_adj_mat(context, mpc.pid)
        # list[int], 每个元素是一个顶点的度
        adj_mat = records.fields['adj_mat']
        assert isinstance(adj_mat, np.ndarray)
        ic(adj_mat)
        deg_list = [secint(int(np.sum(adj_mat[:, i]))) for i in range(V)]  # V
        adj_mat = adj_mat.flatten().tolist()  # 转化为一维list
        adj_mat = [secint(a) for a in adj_mat]  # V * V
        adj_mat = mpc.input(adj_mat)  # (L, V * V)
        ic(len(adj_mat), len(adj_mat[0]))
        adj_mat_flat = seclist(sectype=secint4)  # 将邻接矩阵展平成一维矩阵 L * V * V
        for a in adj_mat:
            adj_mat_flat += a
        ic(len(adj_mat_flat))
        deg_list = mpc.input(deg_list)  # (L, V)
        deg_list_flat = seclist(sectype=secint4)  # L * V
        for a in deg_list:
            deg_list_flat += a
        ic(len(deg_list_flat))
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度列表
            per_id = [deg_list_flat[l * V + i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        ic(await mpc.output(list(I)))
        # core是最终要返回的加密的结果
        core = [[secint(PADDING)] for _ in range(V - 1)]
        ic('根据k逐渐增大来依次淘汰顶点')
        for k in range(V - 1):
            ic(f'---------{k}---------')
            ic('获取I中值等于k的顶点')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, -1)
                              for v_id in range(V)], secint4)
            ic('遍历所有I值等于k的顶点')
            # 后面可能还会有新的顶点加入v_list
            while v_list:
                # 在每一层分别中去掉这个节点
                ic('移除选中的顶点')
                # v_id有可能为0
                v_id = v_list.pop(0)
                core[k].append(v_id)
                # 所有需要修改I[v_id]的节点id
                updated = []
                # ic('获取v_id的邻居, 度--')
                for l in range(L):
                    # 第l层的邻接矩阵
                    for u_id in range(V):
                        # 第l层节点u_id和节点v_id是否是邻居
                        deg_list_flat[l * V + u_id] -= mpc.if_else(
                            (v_id != -1) & (adj_mat_flat[l * V * V + v_id * V + u_id] == 1), 1, 0)
                        updated.append(mpc.if_else(
                            (v_id != -1) & (I[u_id] > k) & (I[u_id] - 1 == deg_list_flat[l * V + u_id]), u_id, PADDING))
                # ic('更新某些邻居节点的I值')
                for u_id in updated:
                    per_id = [deg_list_flat[l * V + u_id] for l in range(L)]
                    # 即使u_id为-1, 该等式也能执行, 只是多了一次无用计算
                    I[u_id] = mpc.if_else(
                        u_id != -1, mpc.sorted(per_id)[-lamb], I[u_id])
        core = [await mpc.output(c) for c in core]
        ic(core)


if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mock_adj_mat2(2))
