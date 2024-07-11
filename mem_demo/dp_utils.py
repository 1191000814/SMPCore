# 差分隐私工具包

import numpy as np
from icecream import ic


def add_lap_noise(data: np.ndarray, epsilon=0, delta=0):
    """
    在给定的数据集添加拉普拉斯噪声
    data: 数据集
    num_nodes: 节点数量, 用来确定选择范围
    epsilon: 隐私预算
    delta: 全局敏感度
    """
    data += np.random.laplace(loc=0, scale=delta / epsilon, size=data.shape)


def add_gauss_noise(data: np.ndarray, epsilon=0, delta=0):
    """
    在给定的数据集添加拉普拉斯噪声
    data: 数据集
    num_nodes: 节点数量, 用来确定选择范围
    epsilon: 隐私预算
    delta: 全局敏感度
    """
    data += np.random.normal(loc=0, scale=delta / epsilon, size=data.shape)


# def exponential(data: np.ndarray, num_nodes, epsilon=0):
#     # 使用指数机制添加噪声
#     # num_nodes: 节点总数量
#     # epsilon: 隐私预算
#     max_d = np.max(data)  # 最大的度
#     bound = min(2 * max_d, num_nodes)  # 全局范围
#     ic(max_d, bound)
#     # 输出每个值的概率, 范围为[0, 2a]
#     for i in range(data.shape[0]):
#         for j in range(data.shape[1]):
#             data[i][j] = add_exp_noise(data[i][j], bound, epsilon, min(2 * max_d, num_nodes))


# def add_exp_noise(raw_val, bound, epsilon=0, delta=0.1):
#     # 当真实值为loc时, a值出现的概率
#     def prob(loc, a): return math.exp(-epsilon * math.fabs(a - loc) /
#                                       delta) if a < bound else 0
#     all_vals = np.arange(bound)
#     probs = np.array([prob(raw_val, val) for val in all_vals])
#     probs = probs / sum(probs)  # 归一化
#     # ic(probs)
#     return np.random.choice(all_vals, p=probs)


def add_exp_noise(data: np.ndarray, num_nodes, epsilon=0, delta=0.1):
    """
    使用指数机制添加噪声
    data: [layer, num_nodes], 每一层的每一个节点的度
    num_nodes: 节点总数量
    epsilon: 隐私预算
    """
    ic(data.shape)
    layers = data.shape[0]
    max_d = np.max(data)  # 最大的度
    # 添加噪声之后, 所有可能的度值出现的范围为[0, bound]
    bound = min(2 * max_d, num_nodes)
    ic(max_d, bound)

    # [layers, num_nodes, bound]
    all_vals = np.tile(np.arange(bound), (layers, num_nodes, 1))  # 按行复制
    ic(all_vals.shape)
    # [layers, num_nodes, bound]
    probs = np.exp(-epsilon * np.abs(all_vals - data[:, :, np.newaxis]) / delta)
    probs[data >= bound] = 0  # 将超出边界的概率设为0
    probs /= np.sum(probs, axis=-1, keepdims=True)  # 归一化
    ic(probs.shape)
    for l in range(layers):
        for v_id in range(num_nodes):
            data[l, v_id] = np.random.choice(all_vals[l][v_id], p=probs[l][v_id])


def loss(raw_d: dict, test_d: dict):
    """
    raw_d, test_d: k -> set(v_id)
    两个字典对应的key的value差值之和 / key总数
    返回[平均每个节点的精度损失了多少]
    """
    loss = 0.0
    # 转化为 v_id -> k
    raw_d_T = {v_id: k for k, v_set in raw_d.items() for v_id in v_set}
    test_d_T = {v_id: k for k, v_set in test_d.items() for v_id in v_set}
    assert len(raw_d) == len(test_d)
    for v_id in range(len(raw_d_T)):
        # 字典的id是从0开始的
        loss += abs(raw_d_T[v_id] - test_d_T[v_id])
    return loss / len(raw_d_T)
