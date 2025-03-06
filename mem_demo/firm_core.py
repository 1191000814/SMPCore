"""
论文firmcore decomposition of multilayer networks的实现
"""
import utils
import numpy as np
from icecream import ic
from tqdm import tqdm
import networkx as nx
import time
import collections
import dataset as DS

def firm_core1(MG0: nx.MultiGraph, num_layers, lamb):
    '''
    先自己创建一个模拟数据集, 用于测试
    返回多层无向图中(lamb, k)-FirmCore的顶点集合
    明文实现, 无隐私设置
    '''
    MG = MG0.copy()
    start_time = time.time()
    num_nodes = MG.number_of_nodes()
    assert lamb <= num_layers
    # 每个顶点vertices对应的Top-λ(deg(vertices))
    I = [-1 for _ in range(num_nodes)]

    # B[i]表示Top-λ(deg(vertices))为i的顶点有哪些, 因为度的最大值只能为num_nodes-1
    # B = [set() for _ in range(num_nodes - 1)]
    B = collections.defaultdict(set)
    # core-λ(vertices)
    # Core[k]表示在第k轮迭代时被移除的点
    core = collections.defaultdict(set)
    # core = np.array([set() for _ in range(num_nodes)])
    # 度矩阵, Degree[l][id]为第l层标识符为id的顶点的度, 会随着每轮迭代变化
    Degree = utils.get_all_degree(MG, num_nodes, num_layers)
    # 初始化数组I和B
    with tqdm(range(num_nodes)) as pbar:
        for i in range(num_nodes):
            # id为i的顶点在各层的第λ大的度
            top_d = np.partition(Degree[:, i], -lamb)[-lamb]
            assert top_d <= num_nodes
            I[i] = top_d
            B[top_d].add(i)
        # ic(I)
        # ic(B)
        # 从0开始依次移除最小Top-d的顶点
        count = 0
        for k in range(num_nodes - 1):
            if count == num_nodes:
                break
            # ic(f'-------------{k}--------------')
            while B[k]:
                v_id = B[k].pop()
                # ic(v_id)
                count += 1
                pbar.update(1)
                core[k].add(v_id)
                # 移除v后, 将需要修改top_d的顶点存储在N中
                N = set()
                # 寻找v在所有层中的邻居(必须是还未被移除的)
                for u_id, layers in MG[v_id].items():
                    for l in layers:
                        Degree[l][u_id] -= 1
                        if I[u_id] > k and Degree[l][u_id] == I[u_id] - 1:
                            N.add(u_id)
                # 更新需要更新的邻居的值
                for u_id in N:
                    # ic(u_id)
                    B[I[u_id]].remove(u_id)
                    I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
                    B[I[u_id]].add(u_id)
                MG.remove_node(v_id)
                I[v_id] = 0
                # ic(I)
                # ic(B)
    print_result(core, start_time)


def firm_core2(MG0: nx.MultiGraph, num_layers, lamb):
    '''
    修改: 每次删除所有能找到I[i]=k的点, 然后整体更新I
    每次直接使用删除后的多层图重新计算I和B
    第k次迭代一直到没有小于或等于k的I值为止
    正确性√ 效率比前一种方法略低
    '''
    MG = MG0.copy()
    start_time = time.time()
    num_nodes = MG.number_of_nodes()
    assert lamb <= num_layers
    core = collections.defaultdict(set)
    _, B, _ = get_IB(MG, num_nodes, 0, lamb)
    # 从0开始依次移除最小Top-d的顶点
    count = 0
    with tqdm(range(num_nodes)) as pbar:
        for k in range(num_nodes):
            if count == num_nodes:
                break
            # ic(f'-------------{k}--------------')
            while B[k]:
                # ic(B)
                # ? 需要删除任何I值[不大于k]的顶点, 而不是等于k的顶点
                core[k] = core[k] | B[k]
                pbar.update(len(B[k]))
                # ic(B[k])
                # 重新计算I和B
                MG.remove_nodes_from(B[k])
                count += len(B[k])
                # ic(MG.number_of_nodes())
                # 更新需要更新的邻居的值
                _, B, _ = get_IB(MG, num_nodes, k, lamb)
    print_result(core, start_time)


def firm_core3(MG0: nx.MultiGraph, num_layers, lamb, switch_num=3):
    '''
    修改: 综合方法1和方法2, 先批量删除, 再逐个删除
    正确性？ 效率介于两种方法之间
    '''
    MG = MG0.copy()
    start_time = time.time()
    num_nodes = MG.number_of_nodes()
    assert lamb <= num_layers
    core = collections.defaultdict(set)
    I, B, _ = get_IB(MG, num_nodes, 0, lamb)
    # 从0开始依次移除最小Top-d的顶点
    count = 0
    with tqdm(range(num_nodes)) as pbar:
        for k in range(num_nodes):
            if count == num_nodes:
                break
            # ic(f'-------------{k}--------------')
            phase = True  # 批量删除
            while B[k]:
                if phase:  # 批量删除
                    # ic(B)
                    # ? 需要删除任何I值[不大于k]的顶点, 而不是等于k的顶点
                    core[k] = core[k] | B[k]
                    pbar.update(len(B[k]))
                    # ic(B[k])
                    # 重新计算I和B
                    MG.remove_nodes_from(B[k])
                    count += len(B[k])
                    # ic(MG.number_of_nodes())
                    # 更新需要更新的邻居的值
                    I, B, Degree = get_IB(MG, num_nodes, k, lamb)
                    if len(B[k]) <= switch_num:
                        phase = False
                else:  # 逐个删除
                    v_id = B[k].pop()
                    # ic(v_id)
                    count += 1
                    pbar.update(1)
                    core[k].add(v_id)
                    # 移除v后, 将需要修改top_d的顶点存储在N中
                    N = set()
                    # 寻找v在所有层中的邻居(必须是还未被移除的)
                    for u_id, layers in MG[v_id].items():
                        for l in layers:
                            Degree[l][u_id] -= 1
                            if I[u_id] > k and Degree[l][u_id] == I[u_id] - 1:
                                N.add(u_id)
                    # 更新需要更新的邻居的值
                    for u_id in N:
                        # ic(u_id)
                        B[I[u_id]].remove(u_id)
                        I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
                        B[I[u_id]].add(u_id)
                    MG.remove_node(v_id)
                    I[v_id] = 0
    print_result(core, start_time)


def print_result(core, start_time):
    run_time = time.time() - start_time
    ic(run_time)
    # total_num = 0
    # for k, nodes in core.items():
    #     total_num += len(nodes)
    #     ic(k, len(nodes))
    # ic(total_num)


def get_IB(MG: nx.MultiGraph, num_nodes, k, lamb):
    I = collections.defaultdict(int)

    # B[i]表示Top-λ(deg(vertices))为i的顶点有哪些, 因为度的最大值只能为num_nodes-1
    B = collections.defaultdict(set)
    # 度矩阵, Degree[l][id]为第l层标识符为id的顶点的度, 会随着每轮迭代变化
    Degree = utils.get_all_degree(MG, num_nodes, num_layers)
    # 初始化数组I和B
    for v_id in MG.nodes:
        # id为i的顶点在各层的第λ大的度
        top_d = np.partition(Degree[:, v_id], -lamb)[-lamb]
        assert top_d <= num_nodes
        I[v_id] = top_d
        B[max(top_d, k)].add(v_id)
    return I, B, Degree


# def update_IB(MG: nx.MultiGraph, I, B, Degree, k, v_id, lamb):
#     '''
#     删除一个节点后, 更新节点的度和部分节点的I值
#     '''
#     # 移除v后, 将需要修改top_d的顶点存储在N中
#     N = set()
#     # 寻找v在所有层中的邻居(必须是还未被移除的)
#     for u_id, layers in MG[v_id].items():
#         for l in layers:
#             Degree[l][u_id] -= 1
#             if I[u_id] > k and Degree[l][u_id] == I[u_id] - 1:
#                 N.add(u_id)
#     # 更新需要更新的邻居的值
#     for u_id in N:
#         # ic(u_id)
#         B[I[u_id]].remove(u_id)
#         I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
#         B[I[u_id]].add(u_id)
#     MG.remove_node(v_id)
#     I[v_id] = 0


if __name__ == '__main__':
    dataset = ['homo', 'sacchcere', 'sanremo', 'slashdot', 'ADHD', 'FAO', 'RM', 'TD']
    # MG = create_data.create_graph()
    MG, num_layers = DS.read_graph(dataset[3], 3)
    ic(num_layers)
    firm_core1(MG, 3, 1)
    firm_core2(MG, 3, 1)
    firm_core3(MG, 3, 1)
    firm_core1(MG, 3, 2)
    firm_core2(MG, 3, 2)
    firm_core3(MG, 3, 2)
    firm_core1(MG, 3, 3)
    firm_core2(MG, 3, 3)
    firm_core3(MG, 3, 3)