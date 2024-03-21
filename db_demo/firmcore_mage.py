import mgp_mock as mgp
import numpy as np
from icecream import ic


@mgp.write_proc
# 移除某一个顶点, 并更新其邻居的度
# 必须要加参数layer, 因为每一方只能移除自己这边的节点
# 移除顶点后, 每层返回涉及的邻居u变化之后的度(如果为I[u]-1, 那么需要更新I[u]), 格式为{节点id: 变化之后的度]}
# 在安全多方计算中, 不能将安全计算得到的[中间结果]发送给任何一方, 所以需要把{修改节点id, 修改后的度}发给安全计算
# ? 这里的节点不能修改? v.underlying_graph_is_mutable()显示为False
# ! 暂时弃用, 因为不再需要设置节点的度属性
def remove_node1(context: mgp.ProcCtx, layer, v_id) -> mgp.Record(updated=mgp.Nullable[dict]):
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        updated = {}
        if int(v.properties['layer']) == layer and int(v.properties['id']) == v_id:
            # 其邻居的度都--
            ic(v.underlying_graph_is_mutable())
            for e in v.in_edges:
                assert isinstance(e, mgp.Edge)
                d = int(e.from_vertex.properties['deg']) - 1
                e.from_vertex.properties.set('deg', d)
                updated[int(e.from_vertex.properties['id'])] = d
            for e in v.out_edges:
                assert isinstance(e, mgp.Edge)
                d = int(e.to_vertex.properties['deg']) - 1
                e.to_vertex.properties.set('deg', d)
                updated[int(e.to_vertex.properties['id'])] = d
            return mgp.Record(updated=updated)
    # 表示出错
    return mgp.Record(updated=updated)


@mgp.write_proc
# 移除某一个顶点, 并更新其邻居的度
# 必须要加参数layer, 因为每一方只能移除自己这边的节点
# 移除顶点后, 每层返回涉及的邻居u变化之后的度(如果为I[u]-1, 仅返回[节点id], 内存中保存变化的整个图的度矩阵
def remove_node(context: mgp.ProcCtx, layer, v_id) -> mgp.Record(updated=mgp.Nullable[set]):
    updated = []
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        if int(v.properties['layer']) == layer and int(v.properties['id']) == v_id:
            # 返回该节点的邻居
            for e in v.in_edges:
                assert isinstance(e, mgp.Edge)
                updated.append(int(e.from_vertex.properties['id']))
            for e in v.out_edges:
                assert isinstance(e, mgp.Edge)
                updated.append(int(e.to_vertex.properties['id']))
            return mgp.Record(updated=updated)
    # 该节点没有邻居, 返回一个空集合
    return mgp.Record(updated=updated)


@mgp.write_proc
# 在算法最开始阶段，我们设置所有节点的度
# 为了简便，我们把所有层一起初始化
# ! 暂时弃用
def init_deg(context: mgp.ProcCtx):
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        v.properties['deg'] = len(list(v.in_edges) + list(v.out_edges))


# 获取某一层某一个节点的度
def get_deg(v: mgp.Vertex) -> mgp.Record(deg=mgp.Nullable[int]):
    return len(list(v.in_edges) + list(v.out_edges))


@mgp.read_proc
# 获取整个图的邻接表
# 每个第i个子数组属于顶点v, 格式为[[节点1的邻居1, 节点1的邻居2], [节点2的邻居1, 节点2的邻居2]...]
def get_adj_list(context: mgp.ProcCtx, layer) -> mgp.Record(adj_list=dict, deg_list=list):
    adj_dict = {}
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        if int(v.properties['layer']) == layer:
            v_id = int(v.properties['id'])
            assert v_id not in adj_dict
            adj_dict[v_id] = [v_id]
            for e in v.in_edges:
                adj_dict[v_id].append(int(e.from_vertex.properties['id']))
            for e in v.out_edges:
                adj_dict[v_id].append(int(e.to_vertex.properties['id']))
    # 每层的顶点数
    num_v = len(adj_dict)
    deg_list = [None for _ in range(num_v)]  # 度列表
    adj_list = [None for _ in range(num_v)]  # 邻接表
    for v_id, neighbors in adj_dict.items():
        adj_list[v_id] = neighbors
        deg_list[v_id] = len(neighbors) - 1  # 减去自己
    return mgp.Record(adj_list=adj_list, deg_list=deg_list)


@mgp.read_proc
# 获取整个图的邻接矩阵
def get_adj_mat(context: mgp.ProcCtx, layer):
    adj_dict = {}
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        if int(v.properties['layer']) == layer:
            v_id = int(v.properties['id'])
            assert v_id not in adj_dict
            adj_dict[v_id] = [v_id]
            for e in v.in_edges:
                adj_dict[v_id].append(int(e.from_vertex.properties['id']))
            for e in v.out_edges:
                adj_dict[v_id].append(int(e.to_vertex.properties['id']))
    num_v = len(adj_dict)
    adj_mat = np.full([num_v, num_v], 0, dtype=int)
    for v_id, neighbors in adj_dict.items():
        for u_id in neighbors:
            adj_mat[v_id, u_id] = 1
    return mgp.Record(adj_mat=adj_mat)


@mgp.read_proc
# 在最开始获取所有节点的度
def get_all_deg(context: mgp.ProcCtx, layer) -> mgp.Record(deg_list=mgp.Nullable[list]):
    deg_dict = {}
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        if int(v.properties['layer']) == layer:
            deg_dict[int(v.properties['id'])] = get_deg(v)
    deg_list = [-1 for _ in range(len(deg_dict))]
    for v_id, d in deg_dict.items():
        deg_list[v_id] = d
    return mgp.Record(deg_list=deg_list)
