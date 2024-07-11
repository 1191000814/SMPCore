# 使用edge-local差分隐私给每个层分别添加噪声, 两个对应顶点相同, 只有一条边对应不同的图视为[相邻图]
# 在本地使用本地差分隐私, 每个图的都使用自己数据集的平滑敏感度(通过局部敏感度计算)作为拉普拉斯噪声的参数
 
import create_data
import utils
import dp_utils
import numpy as np
from icecream import ic
import networkx as nx
import time
import collections

def firm_core(MG: nx.MultiGraph, num_layers, lamb, epsilon):
    """
    先自己创建一个模拟数据集, 用于测试
    返回多层无向图中(lamb, k)-FirmCore的顶点集合
    明文数据+噪声实现, 无加密设置
    """
    t_begin = time.time()
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
    if epsilon > 0:
        ic("add noise")
        dp_utils.add_exp_noise(Degree, MG.number_of_nodes(), epsilon)
    for i in range(num_nodes):
        # id为i的顶点在各层的第λ大的度
        top_d = np.partition(Degree[:, i], -lamb)[-lamb]
        assert top_d <= num_nodes
        I[i] = top_d
        B[top_d].add(i)
    # ic(I)
    # ic(B)
    # 从0开始依次移除最小Top-d的顶点
    for k in range(num_nodes - 1):
        # print(f'-------------{k}--------------')
        while B[k]:
            v_id = B[k].pop()
            ic(v_id)
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
    t_end = time.time()
    ic(t_end - t_begin)
    return core


if __name__ == "__main__":
    # MG = create_data.create_graph()
    MG, num_layers = create_data.create_by_file("homo")
    core_dp = firm_core(MG.copy(), num_layers, 2, 0.2)
    core = firm_core(MG, num_layers, 2, 0)
    ic(core)
    ic(core_dp)
    for k in core.keys():
        ic(k, len(core[k]))
    for k in core_dp.keys():
        ic(k, len(core_dp[k]))
    ic(dp_utils.loss(core, core_dp))
