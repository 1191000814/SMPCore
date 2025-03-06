# 暂时不使用图数据库, 而使用单独的图结构作为各方的数据源np
# 无向图smpc-firmcore算法
# * 计算各种算子的执行次数
# ? 异步函数可以调用非异步函数, 非异步函数不可以调用异步函数
# ? 调用一个函数时, 不需要在被调用的函数中重复声明 'aysnc with mpc:'

import numpy as np
from mpyc.runtime import mpc
from icecream import ic
from time import time
from tqdm import tqdm
import dataset as DS
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
parser.add_argument('--param_lambda', '-l', help='参数λ的值')
args = parser.parse_args()


class FirmCore:

    # __init__不能是异步函数
    def __init__(self, lamb, dataset=None):
        self.layer = mpc.pid  # 该层id
        if dataset is None:  # 测试图数据
            self.G = DS.read_my_layer(self.layer)
        else:  # 真实图数据
            self.G = DS.read_layer(dataset, mpc.pid)
        self.num_layers = len(mpc.parties)  # 层数
        if lamb is not None:
            self.lamb = int(lamb)
        else:
            self.lamb = math.ceil(self.num_layers / 2)  # λ = m // 2
        ic(f'{self.layer + 1} of {self.num_layers} party')
        self.num_nodes = self.G.number_of_nodes()  # 节点数
        self.max_bit_len = self.num_nodes.bit_length()  # 临时确定的bit数, 后面需要改成最大度需要的bit数
        ic(self.max_bit_len)
        self.sec_int_type = mpc.SecInt(self.max_bit_len)
        self.num_sort = dict()  # * {迭代轮数: 排序算子执行次数}
        self.num_comm = dict()  # * {迭代轮数: 通信传输的总数据量}

    def sec_int(self, num):
        return self.sec_int_type(num)

    async def smpcore(self):
        # 原算法1-1
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init B")
            core = collections.defaultdict(set)
            # 从0开始依次移除最小Top-d的顶点
            with tqdm(range(self.num_nodes)) as pbar:
                for k in range(self.num_nodes):
                    self.num_sort[k] = 0
                    self.num_comm[k] = 0
                    if len(remain_v) == 0:
                        break
                    ic(f"--------{k}-------")
                    # ? 和v1不同的地方: 每轮k删除完之后, 重新获取新的I和B
                    B = await self.init_IB(deg_list, remain_v, k=k)
                    while B[k]:
                        v_id = B[k].pop()
                        core[k].add(v_id)
                        remain_v.remove(v_id)
                        pbar.update(1)
                        # 移除v后, 将需要修改I的顶点存储在N中
                        update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                        # ? 需要考虑所有层中一共有哪些节点需要更新
                        for u_id in set(self.G.neighbors(v_id)) & remain_v:
                            deg_list[u_id] -= 1
                            # ? 在这里没法获取I不为k节点具体的I值, 所以我们将当前节点所有[I值不为k]且[未被删除的]邻居都加入update_v
                            # (其实就是I值大于k的邻居节点)
                            if u_id not in B[k]:
                                update_v.add(u_id)
                        await self.update_IB(deg_list, list(update_v), B, k)
        self.print_result(core, start_time)

    async def smpcore_br(self):
        # 原算法2-1
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init B")
            core = collections.defaultdict(set)
            with tqdm(range(self.num_nodes)) as pbar:
                # 从0开始依次移除最小Top-d的顶点
                for k in range(self.num_nodes):
                    self.num_sort[k] = 0
                    self.num_comm[k] = 0
                    if len(remain_v) == 0:
                        break
                    ic(f"--------{k}-------")
                    B = await self.init_IB(deg_list, remain_v, k)
                    while B[k]:
                        assert len(core[k] & B[k]) == 0
                        core[k] = core[k] | B[k]
                        remain_v = remain_v - B[k]
                        pbar.update(len(B[k]))
                        # ic(len(remain_v))
                        # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                        self.G.remove_nodes_from(B[k])
                        deg_list = utils.get_degree(self.G, self.num_nodes)
                        B = await self.init_IB(deg_list, remain_v, k)
        ic(len(core[0]))
        self.print_result(core, start_time)

    async def smpcore_ar(self, switch_num=100):
        # 原算法3-1
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init B")
            core = collections.defaultdict(set)
            with tqdm(range(self.num_nodes)) as pbar:
                for k in range(self.num_nodes):
                    self.num_sort[k] = 0
                    self.num_comm[k] = 0
                    if len(remain_v) == 0:
                        break
                    ic(f"--------{k}-------")
                    B = await self.init_IB(deg_list, remain_v, k)
                    batch_remove = True  # 处于哪一阶段, 批量True/逐个False
                    while B[k]:
                        # 第一阶段: 批量删除节点, 每次重新计算全部节点的I/B
                        if batch_remove:
                            core[k] = core[k] | B[k]
                            remain_v = remain_v - B[k]
                            pbar.update(len(B[k]))
                            # ic(len(remain_v))
                            # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                            self.G.remove_nodes_from(B[k])
                            deg_list = utils.get_degree(self.G, self.num_nodes)
                            B = await self.init_IB(deg_list, remain_v, k)
                            if len(B[k]) > 0 and len(remain_v) / len(B[k]) > switch_num:
                                batch_remove = False
                        # 第二阶段: 逐个删除节点, 每次只重新计算指定节点的I/B
                        else:
                            v_id = B[k].pop()
                            core[k].add(v_id)
                            remain_v.remove(v_id)
                            pbar.update(1)
                            update_v = set()  # ! 本轮需要修改I值的顶点, 每轮都需要更新
                            # ? 需要考虑所有层中一共有哪些节点需要更新
                            for u_id in set(self.G.neighbors(v_id)) & remain_v:
                                deg_list[u_id] -= 1
                                if u_id not in B[k]:
                                    update_v.add(u_id)
                            self.G.remove_node(v_id)
                            await self.update_IB(deg_list, list(update_v), B, k)
        self.print_result(core, start_time)

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

    def print_result(self, core: collections.defaultdict, start_time):
        '''
        算法运行结束时打印一下结果和时间
        '''
        total_sort = 0
        total_comm = 0
        total_num = 0
        for k, nodes in core.items():
            total_num += len(nodes)
            ic(k, len(nodes))
        for num_sort_k in self.num_sort.values():
            total_sort += num_sort_k
        for num_comm_k in self.num_comm.values():
            total_comm += num_comm_k
        ic(self.num_sort)
        ic(self.num_comm)
        ic(total_sort)
        ic(total_comm / self.num_layers)
        run_time = (time() - start_time) / 60
        ic(run_time)
        # assert total_num == self.num_nodes

    async def init_IB(self, deg_list, remain_v, k=0):
        '''
        修改版本2:
        在firm_core_v2中, 每次遍历只用到了B[k]的值, 那么I中值不等于k的值可以不公开计算, 减少隐私泄露量
        计算I的密文时, 如果I[i] > k, 设置I[i] = -1
        '''
        remain_v = list(remain_v)  # 变为列表, 需要保持遍历有序
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        self.num_comm[k] += len(remain_v) * self.num_layers  # * comm
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)
        # remain_v中的原id -> (从0开始的连续)新id
        node2idx = {v_id: i for i, v_id in enumerate(remain_v)}
        Degree = Degree[:, remain_v]  # 先切片
        self.num_sort[k] += len(remain_v)  # * sort
        I0 = np.sort(Degree, axis=0)[-self.lamb, :]  # 再排序
        #! 这里注意I0是否为空
        if len(I0) > 0:
            # mask
            I1 = [mpc.if_else(i > k, pad_val, k) for i in I0]
            I1 = await mpc.output(I1)
        I = [pad_val for _ in range(self.num_nodes)]
        for v_id in remain_v:
            I[v_id] = I1[node2idx[v_id]]
        # ? 既然I是明文的, 那么也可以有B
        B = collections.defaultdict(set)
        for v_id in remain_v:
            # 只记录I值为k的顶点在各层的第λ大的度, 均记为k
            if I[v_id] == k:
                B[k].add(v_id)
        return B

    async def update_IB(self, deg_list, update_v: list, B, k):
        """
        修改版本1, 传入的I为仅有k值的节点, 每次更新数组I, 并每次将I值-1之后为k的节点加入B
        """
        # ? 在mod1的基础上进行修改, 不重新计算所有的I的值, 只计算必须修改I值节点的I值
        deg_list = [self.sec_int(deg) for deg in deg_list]  # 转化成安全矩阵
        # Degree的[行]为层数, [列]为id
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)

        #! 这里需要使得每个mpc.input的数量都相等
        update_nums = mpc.input(self.sec_int(len(update_v)))  # 单层图需要修改I的数量
        max_update_num = await mpc.output(mpc.max(update_nums))  # 各层中最大需要修改的数量
        real_total_num = await mpc.output(mpc.sum(update_nums))  # 各层中所有需要修改的数量之和
        self.num_comm[k] += real_total_num  # * comm
        if max_update_num == 0:  # 传入的长度至少为1
            max_update_num = 1
        update_v.extend([pad_val] * (max_update_num - len(update_v)))  # 填充到最大数量
        all_update_v = mpc.input([self.sec_int(v_id) for v_id in update_v])  # 各层需要改变I的节点
        all_update_v = [v_id for per_layer in all_update_v for v_id in per_layer]  # 全放到一个列表里, 但可能有重复
        all_update_v = set(await mpc.output(all_update_v))  # 明文后用集合表示
        #! pad_val有可能不存在
        all_update_v.discard(pad_val)
        all_update_v = list(all_update_v)  # idx -> v_id
        if all_update_v:
            # 对应all_update_v中的节点
            Degree = Degree[:, all_update_v]
            # 只包含update的顶点
            self.num_sort[k] += len(all_update_v)
            I_update = mpc.np_tolist(np.sort(Degree, axis=0)[-self.lamb, :])
            #! 比update_IB多了这一步
            I0 = [mpc.if_else(i > k, pad_val, k) for i in I_update]
            I0 = await mpc.output(I0)
            for i, v_id in enumerate(all_update_v):
                if I0[i] == k:
                    # 更新B[k]
                    B[k].add(v_id)


if __name__ == "__main__":
    ic(args.dataset, args.version, args.param_lambda, args.switch_num)
    datasets = {1: 'homo', 2: 'sacchcere', 3: 'sanremo', 4: 'slashdot', 5: 'Terrorist', 6: 'RM', 7: 'Yeast', 8: 'Yeast_2'}
    if args.dataset is None:  # 测试数据集
        dataset = None
    elif len(args.dataset) > 1:  # 合成数据集
        dataset = f'synthetic/random_{int(args.dataset)}'
    else:
        dataset_no = int(args.dataset)
        dataset = datasets[dataset_no]
    firm_core = FirmCore(args.param_lambda, dataset)
    if args.version == '1':
        mpc.run(firm_core.smpcore())
    elif args.version == '2':
        mpc.run(firm_core.smpcore_br())
    elif args.version == '3':
        mpc.run(firm_core.smpcore_ar(int(args.switch_num)))
