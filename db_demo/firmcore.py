'''
论文firmcore decomposition of multilayer networks的实现
加入模块中需要将mgp_mock修改为mgp
'''
import mgp_mock as mgp
import nx_graph
import numpy as np
from icecream import ic

# 先自己创建一个模拟数据集, 用于测试
L = 3  # 层数
V = 15  # 每层节点个数


@mgp.read_proc
# 返回多层无向图中(lamb, k)-FirmCore的顶点集合
# 整体数据库实现, 无隐私设置
def firm_core(context: mgp.ProcCtx, lamb) -> mgp.Record(core=mgp.Nullable[list]):

    assert lamb <= L
    # 每个顶点vertices对应的Top-λ(deg(vertices))
    I = [None for _ in range(V)]

    # B[i]表示Top-λ(deg(vertices))为i的顶点有哪些, 因为度的最大值只能为V-1
    B = [set() for _ in range(V - 1)]

    # core-λ(vertices)
    # Core[k]表示在第k轮迭代时被移除的点
    core = [set() for _ in range(V)]

    # Vertices[l][id]为第l层标识符为id的顶点(id从0开始递增)
    Vertices = np.full((L, V), None, mgp.Vertex)
    # 度矩阵, Degree[l][id]为第l层标识符为id的顶点的度, 会随着每轮迭代变化
    Degree = np.full((L, V), None, mgp.Vertex)

    # 遍历一次所有节点
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        # 属性值都为字符串, 转化为整数
        v_id = int(v.properties['id'])
        v_layer = int(v.properties['layer'])
        Vertices[v_layer][v_id] = v
        Degree[v_layer][v_id] = len(list(v.in_edges)) + len(list(v.out_edges))
    ic(Degree)
    # 初始化数组I和B
    for i in range(V):
        # id为i的顶点在各层的第λ大的度
        top_d = np.partition(Degree[:, i], -lamb)[-lamb]
        # ic(top_d)
        assert top_d <= V
        I[i] = top_d
        B[top_d].add(i)

    # 从0开始依次移除最小Top-d的顶点
    for k in range(V - 1):
        while B[k]:
            ic(I)
            ic(B)
            v_id = B[k].pop()
            ic(v_id)
            core[k].add(v_id)
            # 移除v后, 将需要修改top_d的顶点存储在N中
            N = set()
            # 寻找v在所有层中的邻居(必须是还未被移除的)
            for l in range(L):
                v = Vertices[l][v_id]
                assert isinstance(v, mgp.Vertex)
                for e in v.in_edges:
                    u = e.from_vertex
                    u_id = int(u.properties['id'])
                    u_layer = int(u.properties['layer'])
                    Degree[u_layer][u_id] -= 1
                    # 如果v的邻居u度减1前, 等于Top-lambda
                    if I[u_id] > k and Degree[u_layer][u_id] == I[u_id] - 1:
                        N.add(u_id)
                for e in v.out_edges:
                    u = e.to_vertex
                    u_id = int(u.properties['id'])
                    u_layer = int(u.properties['layer'])
                    Degree[u_layer][u_id] -= 1
                    # 还在子图中的点(已经移除的不考虑), 并且度减小之前是第λ大的度
                    if I[u_id] > k and Degree[u_layer][u_id] == I[u_id] - 1:
                        N.add(u_id)
            # 更新需要更新的邻居的值
            for u_id in N:
                ic(u_id)
                B[I[u_id]].remove(u_id)
                I[u_id] = np.partition(Degree[:, u_id], -lamb)[-lamb]
                B[I[u_id]].add(u_id)
    return mgp.Record(core=core)


@mgp.read_proc
# 返回多层有向图中(lamb, k)-FirmD-Core的顶点集合
def firm_d_core(context: mgp.ProcCtx, lamb):
    assert lamb <= L
    # 每个顶点vertices对应的Top_d-和Top_d+
    I = [[None, None] for _ in range(V)]

    # B只用记录入度
    B = [set() for _ in range(V - 1)]

    core = np.full([V - 1, V - 1], set(), dtype=set)

    # Vertices[l][id]为第l层标识符为id的顶点(id从0开始递增)
    Vertices = np.full((L, V), None, mgp.Vertex)
    # 度矩阵, Degree[l][id]为第l层标识符为id的顶点的(入度, 出度), 会随着每轮迭代变化
    Degree = np.full((L, V, 2), -1, int)

    # 遍历一次所有节点
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        # 属性值都为字符串, 转化为整数
        v_id = int(v.properties['id'])
        v_layer = int(v.properties['layer'])
        Vertices[v_layer][v_id] = v
        Degree[v_layer][v_id][0] = len(v.in_edges)  # 入度
        Degree[v_layer][v_id][1] = len(v.out_edges)  # 出度
    ic(Degree)
    # 枚举第一个参数k
    for k in range(V - 1):
        # S: 源节点,所有出度大于k节点id的集合
        # T: 目标节点, 初始化为全部节点id的集合
        # 用u和v表示节点, 方向为 u->v
        S, T = set(), set([i for i in range(V)])
        for v_id in range(V):
            I[v_id][1] = np.partition(
                Vertices[:, int(v.properties['id']), 1], -lamb)[-lamb]
            I[v_id][0] = np.partition(
                Vertices[:, int(v.properties['id']), 0], -lamb)[-lamb]
            # B[k]是T中入度为k的节点的集合
            B[I[v_id, 0]].add(v_id)
            if I[v_id][0] >= k:
                S.add(v_id)
            # 枚举第二个参数r
            for r in range(V - 1):
                while B[r]:
                    v_id = B[r].pop()
                    core[k, r].add(v_id)
                    N = set()
                    # 修改v所有邻居的度
                    for v in Vertices[:, v_id]:
                        assert isinstance(v, mgp.Vertex)
                        # v是T中的节点, 只需要考虑v的上一跳
                        for e in v.in_edges:
                            u = e.from_vertex
                            u_id = int(u.properties['id'])
                            u_layer = int(u.properties['layer'])
                            # 出度--
                            Degree[u_layer][u_id][1] -= 1
                            # 如果v的邻居u度减1前, 等于Top-lambda
                            if I[u_id] > k and Degree[u_layer][u_id][1] == I[u_id] - 1:
                                N.add(u_id)

                        # 需要修改I[u_id][1]值的节点
                        for u_id in N:
                            assert isinstance(u, mgp.Vertex)
                            I[u_id][1] = np.partition(
                                Vertices[:, u_id, 1], -lamb)[-lamb]
                            if I[u_id] < k:
                                S.remove(u_id)
                                for e in u.out_edges:
                                    v = e.to_vertex
                                    v_id = int(u.properties['id'])
                                    v_layer = int(u.properties['layer'])
                                    # 入度--
                                    Degree[v_layer][v_id][0] -= 1
                                    # 还在子图中的点(已经移除的不考虑), 并且度减小之前是第λ大的度
                                    if Degree[v_layer][v_id][0] == I[v_id][0] - 1:
                                        B[I[v_id]].pop(v_id)
                                        I[v_id][0] = np.partition(
                                            Vertices[:, v_id, 0], -lamb)[-lamb]
                                        B[I[v_id]] = I[v_id][0]
    return mgp.Record(core=core)


if __name__ == '__main__':
    G = nx_graph.create_3layer_1()
    context = mgp.ProcCtx(G)
    core = firm_core(context, 2).fields['core']
    ic(core)
    # core_d = firm_d_core(context, 2).fields['core']
    # ic(core_d)