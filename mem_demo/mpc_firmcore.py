# 暂时不使用图数据库, 而使用单独的图结构作为各方的数据源np
# 无向图smpc-firmcore算法
# ? 异步函数可以调用非异步函数, 非异步函数不可以调用异步函数
# ? 调用一个函数时, 不需要在被调用的函数中重复声明 'aysnc with mpc:'

import numpy as np
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from mpyc.statistics import _quickselect
from icecream import ic
from time import time
import create_data
import utils
import collections
import argparse
import math

pad_val = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围
common_lambda = None

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--dataset', '-d', help='选择哪一个数据集(1-4)进行实验')
parser.add_argument('--version', '-v', help='选择哪一种方法(1-2)')
parser.add_argument('--switch_num', '-s', help='当一次迭代批量删除到只剩下几个顶点时, 开始逐个删除')
args = parser.parse_args()


class FirmCore:

    # __init__不能是异步函数
    def __init__(self, lamb, dataset=None):
        self.layer = mpc.pid  # 该层id
        if dataset is None:  # 测试图数据
            self.G = create_data.create_layer(self.layer)
        else:  # 真实图数据
            self.G = create_data.create_layer_by_file(dataset, mpc.pid)
        self.num_layers = len(mpc.parties)  # 层数
        self.lamb = lamb if lamb is not None else math.ceil(self.num_layers / 2)  # λ = m // 2
        ic(f'{self.layer + 1} of {self.num_layers} party')
        self.num_nodes = self.G.number_of_nodes()  # 节点数
        self.max_bit_len = self.num_nodes.bit_length()  # 临时确定的bit数, 后面需要改成最大度需要的bit数
        ic(self.max_bit_len)
        self.sec_int_type = mpc.SecInt(self.max_bit_len)

    def sec_int(self, num):
        return self.sec_int_type(num)

    async def firm_core(self):
        """
        所有的数据都使用密文传输,
        delete B, only use I
        """
        # I: 每个顶点vertices对应的Top-λ(deg(vertices))
        async with mpc:
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
            Degree = np.array(mpc.input(deg_list))  # 整个图的度矩阵
            # Degree的[行]为层数, [列]为id
            ic(len(Degree))
            ic(len(Degree[0]))
            Degree = [seclist(per_layer, self.sec_int_type) for per_layer in Degree]
            # 每个顶点vertices对应的Top-λ(deg(vertices))
            ic("init I")
            I = seclist([pad_val for _ in range(self.num_nodes)], self.sec_int_type)
            for i in range(self.num_nodes):
                # 每个id在各层的度
                per_id = [Degree[l][i] for l in range(self.num_layers)]
                # id为i的顶点在各层的第λ大的度
                I[i] = mpc.sorted(per_id)[-self.lamb]
            # output时必须是list形式
            ic(await mpc.output(list(I)))
            # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
            core = [[self.sec_int(pad_val)] for _ in range(self.num_nodes - 1)]
            k = 0
            counted = 0  # 已经计算了的顶点
            while (k < self.num_nodes) and (counted < self.num_nodes):
                ic(f"---------{k}---------")
                v_list = seclist(
                    [mpc.if_else(I[v_id] == k, v_id, pad_val) for v_id in range(self.num_nodes)],
                    self.sec_int_type,
                )
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
                    self.G.remove_node(v_id0)
                    # 被移除节点的邻居节点集合
                    # ? 直接把所有的度再重新传一遍
                    deg_list = utils.get_degree(self.G, self.num_nodes)
                    deg_list = [self.sec_int(d) for d in deg_list]
                    Degree = np.array(mpc.input(deg_list))
                    # ic(Degree.shape)
                    # Degree的[行]为层数, [列]为id
                    Degree = [seclist(per_layer, self.sec_int_type) for per_layer in Degree]
                    # 每个顶点vertices对应的Top-λ(deg(vertices))
                    # ic('更新数组I')
                    I = seclist([pad_val for _ in range(self.num_nodes)], self.sec_int_type)
                    for i in range(self.num_nodes):
                        # 每个id在各层的度
                        per_id = [Degree[l][i] for l in range(self.num_layers)]
                        # id为i的顶点在各层的第λ大的度
                        # 这里不需要考虑I[i]>k, 因为v_list只多不少
                        I[i] = mpc.sorted(per_id)[-self.lamb]
                    for i in self.G.nodes:
                        v_list.append(mpc.if_else((I[i] == k) & (v_list.count(i) == 0), i, pad_val))
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

    async def firm_core_v1(self, secure=False):
        '''
        修改版本1:
        不再保证整个计算过程中的节点都加密, 仅仅保证每一层最原始的数据(度列表)是加密的, 后续的步骤使用明文计算\n
        迭代过程中, 一个一个地删除迭代B[k]中的节点
        '''
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init I, B")
            I, B = await self.get_IB(deg_list, remain_v)
            core = collections.defaultdict(set)
            # 从0开始依次移除最小Top-d的顶点
            for k in range(self.num_nodes):
                if len(remain_v) == 0:
                    break
                ic(f"--------{k}-------")
                # ic(time() - start_time)
                while B[k]:
                    # ic(I)
                    # ic(B)
                    v_id = B[k].pop()
                    # ic(v_id)
                    # ic(len(remain_v))
                    core[k].add(v_id)
                    remain_v.remove(v_id)
                    # 移除v后, 将需要修改I的顶点存储在N中
                    update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                    # ? 需要考虑所有层中一共有哪些节点需要更新
                    for u_id in set(self.G.neighbors(v_id)) & remain_v:
                        deg_list[u_id] -= 1
                        if deg_list[u_id] == I[u_id] - 1 and I[u_id] > k:
                            update_v.add(u_id)
                    # ic(update_v)
                    await self.update_IB(deg_list, list(update_v), I, B)
                    # ic(time() - start_time)
        total_num = 0
        for k, nodes in core.items():
            total_num += len(nodes)
            ic(k, len(nodes))
        ic(total_num)
        run_time = (time() - start_time) / 60
        ic(run_time)

    async def firm_core_v1_1(self):
        '''
        修改版本1-1:
        在v1的基础上, 每次仅仅泄露当前k时B[k]的值, 即I[v_id]为k的v_id值
        k++时使用get_IB重新获取I值, 迭代中使用update_IB_v1更新I值
        '''
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init I, B")
            core = collections.defaultdict(set)
            # 从0开始依次移除最小Top-d的顶点
            for k in range(self.num_nodes):
                if len(remain_v) == 0:
                    break
                ic(f"--------{k}-------")
                # ic(time() - start_time)
                # ? 和v1不同的地方: 每轮k删除完之后, 重新获取新的I和B
                I, B = await self.get_IB_v2(deg_list, remain_v, k=k)
                while B[k]:
                    # ic(I)
                    # ic(B)
                    v_id = B[k].pop()
                    # ic(v_id)
                    # ic(len(remain_v))
                    core[k].add(v_id)
                    remain_v.remove(v_id)
                    # 移除v后, 将需要修改I的顶点存储在N中
                    update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                    # ? 需要考虑所有层中一共有哪些节点需要更新
                    for u_id in set(self.G.neighbors(v_id)) & remain_v:
                        deg_list[u_id] -= 1
                        # ? 在这里没法获取I不为k节点具体的I值, 所以我们将当前节点所有[I值不为k]且[未被删除的]邻居都加入update_v
                        if I[u_id] == pad_val:
                            update_v.add(u_id)
                    # ic(update_v)
                    await self.update_IB_v1(deg_list, list(update_v), B, k)
                    # ic(time() - start_time)
        total_num = 0
        for k, nodes in core.items():
            total_num += len(nodes)
            ic(k, len(nodes))
        ic(total_num)
        run_time = (time() - start_time) / 60
        ic(run_time)

    async def firm_core_v2(self, secure=False):
        '''
        修改版本2:
        每次迭代删除全部B[k]节点, 然后重新计算I和B
        包括v2和v2-1两个版本, secure: 是否执行get_IB_v2
        '''
        async with mpc:
            start_time = time()
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
                # ic(time() - start_time)
                if secure:
                    _, B = await self.get_IB_v2(deg_list, remain_v, k)
                else:
                    _, B = await self.get_IB(deg_list, remain_v, k)
                while B[k]:
                    # ic(I)
                    # ic(B)
                    # ic(v_id)
                    assert len(core[k] & B[k]) == 0
                    core[k] = core[k] | B[k]
                    remain_v = remain_v - B[k]
                    # ic(len(remain_v))
                    # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                    self.G.remove_nodes_from(B[k])
                    deg_list = utils.get_degree(self.G, self.num_nodes)
                    if secure:
                        _, B = await self.get_IB_v2(deg_list, remain_v, k)
                    else:
                        _, B = await self.get_IB(deg_list, remain_v, k)
                    # ic(time() - start_time)
        total_num = 0
        for k, nodes in core.items():
            total_num += len(nodes)
            ic(k, len(nodes))
        ic(total_num)
        run_time = (time() - start_time) / 60
        ic(run_time)

    async def firm_core_v3(self, secure=False, switch_num=3):
        '''
        修改版本3:
        前面两个版本: v1: 批量删除, v2: 逐个删除
        综合使用版本v1和v2, 对于每一轮k迭代中, 先使用v2批量删除大部分节点, 最后剩余少数节点时, 使用v1逐个删除
        '''
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init I, B")
            core = collections.defaultdict(set)
            # 从0开始依次移除最小Top-d的顶点
            for k in range(self.num_nodes):
                if len(remain_v) == 0:
                    break
                ic(f"--------{k}-------")
                # ic(time() - start_time)
                if secure:
                    I, B = await self.get_IB_v2(deg_list, remain_v, k)
                else:
                    I, B = await self.get_IB(deg_list, remain_v, k)
                phase = True  # 处于哪一阶段, 批量True/逐个False
                while B[k]:
                    # ic(I)
                    # ic(B)
                    # ic(v_id)
                    # 第一阶段: 批量删除节点, 每次重新计算全部节点的I/B
                    if phase:
                        n = len(B[k])
                        core[k] = core[k] | B[k]
                        remain_v = remain_v - B[k]
                        # ic(len(remain_v))
                        # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                        self.G.remove_nodes_from(B[k])
                        deg_list = utils.get_degree(self.G, self.num_nodes)
                        if secure:
                            I, B = await self.get_IB_v2(deg_list, remain_v, k)
                        else:
                            I, B = await self.get_IB(deg_list, remain_v, k)
                        # ic(time() - start_time)
                        if n <= switch_num:
                            phase = False
                    # 第二阶段: 逐个删除节点, 每次只重新计算指定节点的I/B
                    else:
                        v_id = B[k].pop()
                        # ic(v_id)
                        # ic(len(remain_v))
                        core[k].add(v_id)
                        remain_v.remove(v_id)
                        # 移除v后, 将需要修改top_d的顶点存储在N中
                        update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                        # ? 需要考虑所有层中一共有哪些节点需要更新
                        for u_id in set(self.G.neighbors(v_id)) & remain_v:
                            deg_list[u_id] -= 1
                            if I[u_id] == pad_val:
                                update_v.add(u_id)
                        # ic(update_v)
                        if secure:
                            await self.update_IB_v1(deg_list, list(update_v), B, k)
                        else:
                            await self.update_IB(deg_list, list(update_v), I, B)
                        # ic(time() - start_time)
                        pass
        total_num = 0
        for k, nodes in core.items():
            total_num += len(nodes)
            ic(k, len(nodes))
        ic(total_num)
        run_time = (time() - start_time) / 60
        ic(run_time)

    # 下面是工具函数

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

    async def get_IB(self, deg_list, remain_v, k=0):
        """
        初始化I和B
        根据当前层的度矩阵, 获取整个多层图的B结构
        k: 仅在mod2中使用, 即使度小于k也在B中存储在B[k]中
        将mpc.sorted()换成更快的np.sort()函数
        """
        # ic("get local degree list")
        remain_v = list(remain_v)  # 变为列表, 需要保持遍历有序
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        # ic("get all degree list")
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)
        # 原id -> (从0开始的连续)新id
        node2idx = {v_id: i for i, v_id in enumerate(remain_v)}
        # I0 = [pad_val for _ in range(len(remain_v))]
        # ic("compute I")
        Degree = np.sort(Degree, axis=0)
        I0 = [Degree[-self.lamb][v_id] for v_id in remain_v]
        #! 这里注意I0是否为空
        if len(I0) > 0:
            I0 = await mpc.output(I0)
        I = [pad_val for _ in range(self.num_nodes)]
        for v_id in remain_v:
            I[v_id] = I0[node2idx[v_id]]
        # ? 既然I是明文的, 那么也可以有B
        B = collections.defaultdict(set)
        # 只计算剩余节点, 添加进B
        for v_id in remain_v:
            # id为i的顶点在各层的第λ大的度
            B[max(I[v_id], k)].add(v_id)
        return I, B

    async def get_IB_v1(self, deg_list, remain_v, k=0):
        """
        修改版本1: 每次不是求出所有的加密数组再decode, 而是每求出一个数字就写入到明文数组I中
        不用再将remain_id映射到连续整数了
        """
        # ! 因为每次明文都需要等待密文排序全部完成, 所以肯定不如上面函数
        # ! 这样的优化没什么实际意义
        ic("get local degree list")
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        ic("get all degree list")
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)
        I = [pad_val for _ in range(self.num_nodes)]
        ic("compute and get plaintext I")
        for v_id in remain_v:
            # 每个id在各层的度
            per_id = Degree[:][v_id]
            # id为i的顶点在各层的第λ大的度
            deg_lamb = np.sort(per_id)[-self.lamb]
            I[v_id] = await mpc.output(deg_lamb)
        # ? 既然I是明文的, 那么也可以有B
        B = collections.defaultdict(set)
        # 只计算剩余节点, 添加进B
        ic("compute B")
        for v_id in remain_v:
            # id为i的顶点在各层的第λ大的度
            B[max(I[v_id], k)].add(v_id)
        return I, B

    async def get_IB_v2(self, deg_list, remain_v, k=0):
        '''
        修改版本2:
        在firm_core_v2中, 每次遍历只用到了B[k]的值, 那么I中值不等于k的值可以不公开计算, 减少隐私泄露量
        计算I的密文时, 如果I[i] > k, 设置I[i] = -1
        '''
        # ic("get local degree list")
        remain_v = list(remain_v)  # 变为列表, 需要保持遍历有序
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        # ic("get all degree list")
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)
        # 原id -> (从0开始的连续)新id
        node2idx = {v_id: i for i, v_id in enumerate(remain_v)}
        Degree = np.sort(Degree, axis=0)
        I0 = [Degree[-self.lamb][v_id] for v_id in remain_v]
        #! 这里注意I0是否为空
        if len(I0) > 0:
            #! 比get_IB_v1多加了这一步
            I1 = [mpc.if_else(i > k, pad_val, k) for i in I0]
            I1 = await mpc.output(I1)
        I = [pad_val for _ in range(self.num_nodes)]
        for v_id in remain_v:
            I[v_id] = I1[node2idx[v_id]]
        # ? 既然I是明文的, 那么也可以有B
        B = collections.defaultdict(set)
        # 只计算剩余节点, 添加进B
        for v_id in remain_v:
            # id为i的顶点在各层的第λ大的度
            if I[v_id] != pad_val:
                B[I[v_id]].add(v_id)
        return I, B

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
        node2idx = {v_id: i for i, v_id in enumerate(remain_v)}
        I0 = seclist([pad_val for _ in range(len(remain_v))], self.sec_int_type)
        nodes_k = seclist([], self.sec_int_type)
        ic("compute I")
        # ? 和v1一样先计算全部的密文I
        for v_id in remain_v:
            # 每个id在各层的度
            per_id = [Degree[l][v_id] for l in range(self.num_layers)]
            # id为i的顶点在各层的第λ大的度
            I0[v_id] = await mpc.output(mpc.sorted(per_id)[-self.lamb])
        count_k = await mpc.output(I0.count(k))
        ic("select v_id | I[v_id] == k")
        for v_id, I0_k in enumerate(I0):
            nodes_k.append(mpc.if_else(I0_k == k, v_id, pad_val))
        nodes_k.sort(reverse=True)
        del nodes_k[count_k:]
        # ? 明文计算I中的内容
        ic("get plaintext node_k")
        nodes_k = await mpc.output(nodes_k)
        # 新id -> 原id
        nodes_k = [node2idx[i] for i in nodes_k]
        B = collections.defaultdict(set)
        B[k] = set(nodes_k)
        return [], B

    async def update_IB(self, deg_list, update_v: list, I, B):
        """
        根据当前层的度矩阵, 更新多层图的I和B结构
        只计算需要更新的节点, 尽量减少decode操作
        """
        # ? 在mod1的基础上进行修改, 不重新计算所有的I的值, 只计算必须修改I值节点的I值
        # ic("get local degree list")
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        Degree = mpc.input(deg_list)  # [[layer_1], [layer_2]...[layer_n]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array_1, np.array_2...]
        Degree = np.vstack(Degree)
        #! 这里需要使得每个mpc.input的数量都相等
        update_nums = mpc.input(self.sec_int(len(update_v)))  # 单层图需要修改I的数量
        max_update_num = await mpc.output(mpc.max(update_nums))  # 各层中最大需要修改的数量
        if max_update_num == 0:  # 传入的长度至少为1
            max_update_num = 1
        update_v.extend([pad_val] * (max_update_num - len(update_v)))  # 填充到最大数量
        all_update_v = mpc.input([self.sec_int(v_id) for v_id in update_v])  # 各层需要改变I的节点
        # ic(len(all_update_v))
        # ic(all_update_v)
        all_update_v = [v_id for per_layer in all_update_v for v_id in per_layer]  # 全放到一个列表里, 但可能有重复
        all_update_v = set(await mpc.output(all_update_v))  # 明文后用集合表示
        #! pad_val有可能不存在
        all_update_v.discard(pad_val)
        all_update_v = list(all_update_v)  # idx -> v_id
        # ic(all_update_v)
        # ic("update I and B")
        if all_update_v:
            I_update = mpc.np_tolist(np.sort(Degree[:, all_update_v], axis=0)[-self.lamb, :])
            I_update = await mpc.output(I_update, receivers=None)
            # ? 明文计算I中的内容
            for i, v_id in enumerate(all_update_v):
                # id为i的顶点在各层的第λ大的度
                B[I[v_id]].remove(v_id)
                I[v_id] = I_update[i]
                B[I[v_id]].add(v_id)

    async def update_IB_v1(self, deg_list, update_v: list, B, k):
        """
        修改版本1, 传入的I为仅有k值的节点, 每次更新数组I, 并每次将I值-1之后为k的节点加入B
        """
        # ? 在mod1的基础上进行修改, 不重新计算所有的I的值, 只计算必须修改I值节点的I值
        # ic("get local degree list")
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)
        #! 这里需要使得每个mpc.input的数量都相等
        update_nums = mpc.input(self.sec_int(len(update_v)))  # 单层图需要修改I的数量
        max_update_num = await mpc.output(mpc.max(update_nums))  # 各层中最大需要修改的数量
        if max_update_num == 0:  # 传入的长度至少为1
            max_update_num = 1
        update_v.extend([pad_val] * (max_update_num - len(update_v)))  # 填充到最大数量
        all_update_v = mpc.input([self.sec_int(v_id) for v_id in update_v])  # 各层需要改变I的节点
        # ic(len(all_update_v))
        # ic(all_update_v)
        all_update_v = [v_id for per_layer in all_update_v for v_id in per_layer]  # 全放到一个列表里, 但可能有重复
        all_update_v = set(await mpc.output(all_update_v))  # 明文后用集合表示
        #! pad_val有可能不存在
        all_update_v.discard(pad_val)
        all_update_v = list(all_update_v)  # idx -> v_id
        # ic("update I and B")
        if all_update_v:
            # 对应all_update_v中的节点
            I_update = mpc.np_tolist(np.sort(Degree[:, all_update_v], axis=0)[-self.lamb, :])
            I0 = [mpc.if_else(i > k, pad_val, i) for i in I_update]
            I = await mpc.output(I0)
            for i, v_id in enumerate(all_update_v):
                if I[i] != pad_val:
                    # 更新B[k]
                    B[k].add(v_id)


if __name__ == "__main__":
    ic(args.dataset, args.version, args.switch_num)
    if args.dataset[0] == '0':  # 合成数据集
        dataset = f'synthetic/random_{args.dataset[1]}'
    elif args.dataset == '1':
        dataset = 'homo'
    elif args.dataset == '2':
        dataset = 'sacchcere'
    elif args.dataset == '3':
        dataset = 'sanremo'
    elif args.dataset == '4':
        dataset = 'slashdot'
    elif args.dataset == '5':
        dataset = 'ADHD'
    elif args.dataset == '6':
        dataset = 'FAO'
    elif args.dataset == '7':
        dataset = 'RM'
    else:  # 自制测试数据集
        dataset = None
    if args.version == '1':
        mpc.run(FirmCore(common_lambda, dataset).firm_core_v1())
    elif args.version == '1-1':
        mpc.run(FirmCore(common_lambda, dataset).firm_core_v1_1())
    elif args.version == '2':
        mpc.run(FirmCore(common_lambda, dataset).firm_core_v2(False))
    elif args.version == '2-1':
        mpc.run(FirmCore(common_lambda, dataset).firm_core_v2(True))
    elif args.version == '3':
        mpc.run(FirmCore(common_lambda, dataset).firm_core_v3(False, int(args.switch_num)))
    elif args.version == '3-1':
        mpc.run(FirmCore(common_lambda, dataset).firm_core_v3(True, int(args.switch_num)))
