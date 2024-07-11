# 暂时不使用图数据库, 而使用单独的图结构作为各方的数据源np
# 无向图-smpc-firmcore算法
# ? 异步函数可以调用非异步函数, 非异步函数不可以调用异步函数
# ? 调用一个函数时, 不需要在被调用的函数中重复声明 'aysnc with mpc:'

import numpy as np
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


class FirmCore:

    # __init__不能是异步函数
    def __init__(self, lamb, dataset=None):
        self.layer = mpc.pid  # 该层id
        if dataset is None:  # 测试图数据
            self.G = create_data.create_layer(self.layer)
        else:  # 真实图数据
            self.G = create_data.create_layer_by_file(dataset, mpc.pid)
        self.lamb = lamb
        self.num_layers = len(mpc.parties)  # 层数
        ic(f'{self.layer + 1} of {self.num_layers} party')
        self.num_nodes = self.G.number_of_nodes()  # 节点数
        self.max_bit_len = self.num_nodes.bit_length()  # 临时确定的bit数, 后面需要改成最大度需要的bit数
        ic(self.max_bit_len)
        self.sec_int_type = mpc.SecInt(self.max_bit_len)

    def sec_int(self, num):
        return self.sec_int_type(num)

    async def firm_core_v2(self):
        '''
        修改版本2:
        每次删除全部B[k]节点, 然后重新计算I和B
        '''
        async with mpc:
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init I, B")
            # _, B = await self.get_IB_v2(deg_list, remain_v)
            core = collections.defaultdict(set)
            # 从0开始依次移除最小Top-d的顶点
            for k in range(self.num_nodes):
                if len(remain_v) == 0:
                    break
                ic(f"--------{k}-------")
                _, B = await self.get_IB_v3(deg_list, remain_v, k)
                while B[k]:
                    # ic(I)
                    # ic(B)
                    # ic(v_id)
                    core[k] = core[k] | B[k]
                    remain_v = remain_v - B[k]
                    ic(remain_v)
                    # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                    self.G.remove_nodes_from(B[k])
                    deg_list = utils.get_degree(self.G, self.num_nodes)
                    # ic(deg_list)
                    _, B = await self.get_IB_v3(deg_list, remain_v, k)
        for k, nodes in core.items():
            ic(k, len(nodes))

    async def set_sec_bit(self, deg_list: list):
        '''
        设置安全类型bit数, (根据全局最大的度)
        '''
        # 获取全局最大的度
        max_d = self.sec_int(max(deg_list))
        max_d_list = mpc.input(max_d)
        max_d_global: int = await mpc.output(mpc.max(max_d_list))
        ic(max_d_global)
        # 设置安全bit宽度
        self.max_bit_len = self.num_nodes.bit_length()
        # self.max_bit_len = max_d_global.bit_length()  # 修改安全整数bit数
        ic(self.max_bit_len)
        self.sec_int_type = mpc.SecInt(self.max_bit_len)

    async def get_IB_v3(self, deg_list, remain_v, k=0):
        '''
        修改版本3:
        思想同v2, 但是先计算全部I值, 再处理I[v_id]=k的v_id
        '''
        ic("get local degree list")
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        ic("get all degree list")
        Degree = np.array(mpc.input(deg_list))  # 整个图的度矩阵
        # 原id -> 新(连续)id
        id2seq = {v_id: i for i, v_id in enumerate(remain_v)}
        # 新(连续)id -> 原id
        seq2id = {i: v_id for i, v_id in enumerate(remain_v)}
        I_seq = seclist([pad_val for _ in range(len(remain_v))], self.sec_int_type)
        ic("compute I")
        # ? 和v1一样先计算全部的密文I
        for v_id in remain_v:
            # 每个id在各层的度
            v_seq = id2seq[v_id]
            per_id = [Degree[l][v_id] for l in range(self.num_layers)]
            # id为i的顶点在各层的第λ大的度
            I_seq[v_seq] = mpc.sorted(per_id)[-self.lamb]
        # ic(await mpc.output(list(I_seq)))
        ic("select v_seq | I[v_seq] <= k")
        # 这个循环可以隐式表示
        nodes_k = seclist([], self.sec_int_type)
        for v_seq, I_k in enumerate(I_seq):
            nodes_k.append(mpc.if_else(I_k <= k, v_seq, pad_val))
        #! nodes_k可能为空
        if len(nodes_k) > 0:
            count_pad = await mpc.output(nodes_k.count(pad_val))
            nodes_k.sort()
            del nodes_k[:count_pad]
        # ? 明文计算I中的内容
        ic("get plaintext node_k")
        ic(len(nodes_k))
        nodes_k = await mpc.output(list(nodes_k))
        # 新id -> 原id
        nodes_k = [seq2id[i] for i in nodes_k]
        # ic(nodes_k)
        B = collections.defaultdict(set)
        B[k] = set(nodes_k)
        return [], B


if __name__ == "__main__":
    core = mpc.run(FirmCore(2, 'sacchcere').firm_core_v2())
