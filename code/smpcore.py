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

pad_val = -1
common_lambda = None

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--dataset', '-d', help='选择哪一个数据集(1-4)进行实验')
parser.add_argument('--version', '-v', help='选择哪一种方法(1-2)')
parser.add_argument('--switch_num', '-s', help='当一次迭代批量删除到只剩下几个顶点时, 开始逐个删除')
parser.add_argument('--param_lambda', '-l', help='参数λ的值')
args = parser.parse_args()


class FirmCore:
    def __init__(self, lamb, dataset=None):
        self.layer = mpc.pid
        if dataset is None:
            self.G = DS.read_my_layer(self.layer)
        else:
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

    def sec_int(self, num):
        return self.sec_int_type(num)

    async def smpcore(self):
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init B")
            core = collections.defaultdict(set)
            with tqdm(range(self.num_nodes)) as pbar:
                for k in range(self.num_nodes):
                    if len(remain_v) == 0:
                        break
                    ic(f"--------{k}-------")
                    B = await self.init_IB(deg_list, remain_v, k=k)
                    while B[k]:
                        v_id = B[k].pop()
                        core[k].add(v_id)
                        remain_v.remove(v_id)
                        pbar.update(1)
                        update_v = set()
                        # ? 需要考虑所有层中一共有哪些节点需要更新
                        for u_id in set(self.G.neighbors(v_id)) & remain_v:
                            deg_list[u_id] -= 1
                            if u_id not in B[k]:
                                update_v.add(u_id)
                        await self.update_IB(deg_list, list(update_v), B, k)
        self.print_result(core, start_time)

    async def smpcore_br(self):
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)  # 该层的节点的(明文)度列表
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])  # 还没被移除的点
            ic("init B")
            core = collections.defaultdict(set)
            with tqdm(range(self.num_nodes)) as pbar:
                for k in range(self.num_nodes):
                    if len(remain_v) == 0:
                        break
                    ic(f"--------{k}-------")
                    B = await self.init_IB(deg_list, remain_v, k)
                    while B[k]:
                        assert len(core[k] & B[k]) == 0
                        core[k] = core[k] | B[k]
                        remain_v = remain_v - B[k]
                        pbar.update(len(B[k]))
                        # ? 修改本层的度后, 传递给其他方, 然后直接计算新的I和B, 不考虑哪些节点需要更新I与否
                        self.G.remove_nodes_from(B[k])
                        deg_list = utils.get_degree(self.G, self.num_nodes)
                        B = await self.init_IB(deg_list, remain_v, k)
        ic(len(core[0]))
        self.print_result(core, start_time)

    async def smpcore_ar(self, switch_num=100):
        async with mpc:
            start_time = time()
            deg_list = utils.get_degree(self.G, self.num_nodes)
            await self.set_sec_bit(deg_list)
            remain_v = set([i for i in range(self.num_nodes)])
            ic("init B")
            core = collections.defaultdict(set)
            with tqdm(range(self.num_nodes)) as pbar:
                for k in range(self.num_nodes):
                    if len(remain_v) == 0:
                        break
                    ic(f"--------{k}-------")
                    B = await self.init_IB(deg_list, remain_v, k)
                    batch_remove = True
                    while B[k]:
                        if batch_remove:
                            core[k] = core[k] | B[k]
                            remain_v = remain_v - B[k]
                            pbar.update(len(B[k]))
                            self.G.remove_nodes_from(B[k])
                            deg_list = utils.get_degree(self.G, self.num_nodes)
                            B = await self.init_IB(deg_list, remain_v, k)
                            if len(B[k]) > 0 and len(remain_v) / len(B[k]) > switch_num:
                                batch_remove = False
                        else:
                            v_id = B[k].pop()
                            core[k].add(v_id)
                            remain_v.remove(v_id)
                            pbar.update(1)
                            update_v = set()
                            for u_id in set(self.G.neighbors(v_id)) & remain_v:
                                deg_list[u_id] -= 1
                                if u_id not in B[k]:
                                    update_v.add(u_id)
                            self.G.remove_node(v_id)
                            await self.update_IB(deg_list, list(update_v), B, k)
        self.print_result(core, start_time)


    async def set_sec_bit(self, deg_list: list):
        max_d = self.sec_int(max(deg_list))
        max_d_list = mpc.input(max_d)
        max_d_global: int = await mpc.output(mpc.max(max_d_list))
        ic(max_d_global)
        self.max_bit_len = self.num_nodes.bit_length()
        # self.max_bit_len = max_d_global.bit_length()
        ic(self.max_bit_len)
        self.sec_int_type = mpc.SecInt(self.max_bit_len)

    def print_result(self, core: collections.defaultdict, start_time):
        total_num = 0
        # for k, nodes in core.items():
        #     total_num += len(nodes)
        #     ic(k, len(nodes))
        ic(total_num)
        run_time = (time() - start_time) / 60
        ic(run_time)
        # assert total_num == self.num_nodes

    async def init_IB(self, deg_list, remain_v, k=0):
        remain_v = list(remain_v)
        deg_list = [self.sec_int(deg) for deg in deg_list]
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)
        node2idx = {v_id: i for i, v_id in enumerate(remain_v)}
        Degree = Degree[:, remain_v]
        I0 = np.sort(Degree, axis=0)[-self.lamb, :]
        if len(I0) > 0:
            # mask
            I1 = [mpc.if_else(i > k, pad_val, k) for i in I0]
            I1 = await mpc.output(I1)
        I = [pad_val for _ in range(self.num_nodes)]
        for v_id in remain_v:
            I[v_id] = I1[node2idx[v_id]]
        B = collections.defaultdict(set)
        for v_id in remain_v:
            if I[v_id] == k:
                B[k].add(v_id)
        return B

    async def update_IB(self, deg_list, update_v: list, B, k):
        deg_list = [self.sec_int(deg) for deg in deg_list]
        Degree = mpc.input(deg_list)  # [[layer1], [layer2]...[layern]]
        Degree = [mpc.np_fromlist(deg_layer) for deg_layer in Degree]  # [np.array1, np.array2...]
        Degree = np.vstack(Degree)

        update_nums = mpc.input(self.sec_int(len(update_v)))
        max_update_num = await mpc.output(mpc.max(update_nums))
        if max_update_num == 0:
            max_update_num = 1
        update_v.extend([pad_val] * (max_update_num - len(update_v)))
        all_update_v = mpc.input([self.sec_int(v_id) for v_id in update_v])
        all_update_v = [v_id for per_layer in all_update_v for v_id in per_layer]
        all_update_v = set(await mpc.output(all_update_v))
        all_update_v.discard(pad_val)
        all_update_v = list(all_update_v)
        if all_update_v:
            Degree = Degree[:, all_update_v]
            I_update = mpc.np_tolist(np.sort(Degree, axis=0)[-self.lamb, :])
            I0 = [mpc.if_else(i > k, pad_val, k) for i in I_update]
            I0 = await mpc.output(I0)
            for i, v_id in enumerate(all_update_v):
                if I0[i] == k:
                    B[k].add(v_id)


if __name__ == "__main__":
    ic(args.dataset, args.version, args.param_lambda, args.switch_num)
    datasets = {1: 'homo', 2: 'sacchcere', 3: 'sanremo', 4: 'slashdot', 5: 'Terrorist', 6: 'RM', 7: 'Yeast', 8: 'Yeast_2'}
    if args.dataset is None:
        dataset = None
    elif len(args.dataset) > 1:
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
