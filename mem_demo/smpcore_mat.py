from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
import mem_demo.dataset as dataset
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


async def firmcore(lamb):
    # 修改版本, 删去数据结构B, 仅仅使用I
    # 每一方传入的数据是整个图的[邻接矩阵], 之后不再传入其他数据
    # 除了生成v_list, 其他地方尽量不使用明文
    async with mpc:
        l0 = mpc.pid  # 当前层数
        G = dataset.create_layer(l0)
        num_nodes = G.number_of_nodes()
        num_layers = len(mpc.parties)
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {num_layers} party')
        # 获取整个多层图的邻接矩阵, 且压缩为一维向量
        adj_mat = utils.get_adj_mat(G, num_nodes).flatten().tolist()
        # ic(adj_mat)
        adj_mat = [secint(a) for a in adj_mat]  # [num_nodes * num_nodes, ]
        adj_mat = mpc.input(adj_mat)  # [num_layers, num_nodes * num_nodes]
        ic(len(adj_mat), len(adj_mat[0]))
        adj_mat = [seclist(adj_mat_l, secint4) for adj_mat_l in adj_mat]
        # [num_layers, num_nodes * num_nodes]
        # ? deg_list列表不使用多方输入, 用每次更新后的整体邻接矩阵计算
        deg_list = [seclist([sum(adj_mat[l][num_nodes * i: num_nodes * (i + 1)])
                             for i in range(num_nodes)])
                    for l in range(num_layers)]
        I = seclist([pad_val for _ in range(num_nodes)], secint4)
        ic('初始化数组I')
        for i in range(num_nodes):
            # 每个id在各层的度列表
            per_id = [deg_list[l][i] for l in range(num_layers)]
            I[i] = mpc.sorted(per_id)[-lamb]  # id为i的顶点在各层的第λ大的度
        ic(await mpc.output(list(I)))
        core = [[secint(pad_val)] for _ in range(num_nodes)]  # 最终要返回的加密的结果
        # 还未被移除的v
        # remain_v = seclist([i for i in range(num_nodes)], secint4)
        k = 0
        counted = 0
        while k < num_nodes and counted < num_nodes:
            ic(f'---------{k}---------')
            ic('获取I中值等于k的顶点')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, pad_val)
                              for v_id in range(num_nodes)], secint)
            v_list.sort()
            n = await mpc.output(v_list.count(pad_val))
            ic(n)
            del v_list[:n]
            # ic(v_list)
            ic('遍历所有I值等于k的顶点')
            # 后面可能还会有新的顶点加入v_list
            while v_list:
                # 在每一层中都去掉这个节点
                v_id = v_list.pop(0)
                counted += 1
                # await remain_v.remove(v_id)
                core[k].append(v_id)
                # 在邻接矩阵将于点v_id有关的行和列全部置为0
                for l in range(num_layers):
                    for i in range(num_nodes):
                        adj_mat[l][v_id * num_nodes + i] = 0  # v_id所在行全置为0
                        adj_mat[l][i * num_nodes + v_id] = 0  # v_id所在列全置为0
                deg_list = [seclist([sum(adj_mat[l][num_nodes * i: num_nodes * (i + 1)])
                                     for i in range(num_nodes)]) for l in range(num_layers)]
                for v_id in range(num_nodes):  # 每个id在各层的度列表
                    per_id = [deg_list[l][v_id] for l in range(num_layers)]
                    # 更新I, id为i的顶点在各层的第λ大的度
                    I[v_id] = mpc.sorted(per_id)[-lamb]
                    ic(v_id)
                    # ic(await mpc.output(I[k]))
                    # ic(await mpc.output(v_list.count(v_id)))
                    # ic(await mpc.output((I[v_id] == k) & (v_list.count(v_id) == 0)))
                    v_list.append(mpc.if_else((I[v_id] == k) & (
                        v_list.count(v_id) == 0), v_id, pad_val))
                ic(await mpc.output(list(I)))
                n = await mpc.output(v_list.count(pad_val))
                v_list.sort()
                del v_list[:n]
                v_list_out = await mpc.output(list(v_list))
                ic(v_list_out)
            k += 1
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firmcore_mod1(lamb, dataset):
    # 每一方传入的数据是整个图的[邻接矩阵], 之后不再传入其他数据
    # ? 明文计算I和B
    async with mpc:
        l0 = mpc.pid  # layer id
        if dataset is None:
            G = dataset.create_layer(l0)  # 该计算方的单层图
        else:
            G = dataset.create_layer_by_file(dataset, mpc.pid)
        num_layers = len(mpc.parties)  # nums of layers
        num_nodes = G.number_of_nodes()  # nums of nodes
        ic(f'{l0 + 1} of {num_layers} party')
        # the adjacency matrix of the simple graph: [num_nodes * num_nodes]
        adj_mat = utils.get_adj_mat(G, num_nodes).flatten().tolist()
        # ic(adj_mat)
        # encode
        adj_mat = [secint(a) for a in adj_mat]  # [num_nodes * num_nodes, ]
        # the adjacency matrix of the multi-layer graph: [num_layers, num_nodes * num_nodes]
        adj_mat = mpc.input(adj_mat)
        ic(len(adj_mat), len(adj_mat[0]))
        # convert to seclist : [num_layers, num_nodes * num_nodes]
        adj_mat = [seclist(adj_mat[l]) for l in range(num_layers)]
        # didn't input the deg_list, compute from the adj_mat
        deg_list = [seclist([sum(adj_mat[l][num_nodes * i: num_nodes * (i + 1)])
                             for i in range(num_nodes)]) for l in range(num_layers)]
        I = [pad_val for _ in range(num_nodes)]
        ic('init I and B')
        for v_id in range(num_nodes):
            # 每个id在各层的度列表
            per_id = [deg_list[l][v_id] for l in range(num_layers)]
            I[v_id] = mpc.sorted(per_id)[-lamb]  # id为i的顶点在各层的第λ大的度
        ic('get plaintext I')
        I = await mpc.output(I)
        # ? 既然I是明文的, 那么也可以有B
        B = collections.defaultdict(set)
        # 只计算剩余节点, 添加进B
        ic('compute B')
        # the nodes not to be selected
        remain_v = set([v_id for v_id in range(num_nodes)])
        for v_id in remain_v:
            B[I[v_id]].add(v_id)
        core = collections.defaultdict(set)
        for k in range(num_nodes):  # k = 1,2...n-1
            ic(f'--------{k}------')
            while B[k]:
                # ic(I)
                # ic(B)
                v_id = B[k].pop()
                # ic(v_id)
                ic(len(remain_v))
                core[k].add(v_id)
                remain_v.remove(v_id)
                update_v = set()
                for u_id in set(G.neighbors(v_id)) & remain_v:
                    for l in range(num_layers):
                        adj_mat = 1
        ic(core)


if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mod1(2, dataset=None))
