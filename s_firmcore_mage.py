import mgp_mock as mgp
from icecream import ic

V = 15
L = 3


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
                d = int(e.from_vertex.properties['degree']) - 1
                e.from_vertex.properties.set('degree', d)
                updated[int(e.from_vertex.properties['id'])] = d
            for e in v.out_edges:
                assert isinstance(e, mgp.Edge)
                d = int(e.to_vertex.properties['degree']) - 1
                e.to_vertex.properties.set('degree', d)
                updated[int(e.to_vertex.properties['id'])] = d
            return mgp.Record(updated=updated)
    # 表示出错
    return mgp.Record(updated=updated)


@mgp.write_proc
# 移除某一个顶点, 并更新其邻居的度
# 必须要加参数layer, 因为每一方只能移除自己这边的节点
# 移除顶点后, 每层返回涉及的邻居u变化之后的度(如果为I[u]-1, 仅返回[节点id], 内存中保存变化的整个图的度矩阵
def remove_node(context: mgp.ProcCtx, layer, v_id) -> mgp.Record(updated=mgp.Nullable[set]):
    updated = set()
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        if int(v.properties['layer']) == layer and int(v.properties['id']) == v_id:
            # 返回该节点的邻居
            for e in v.in_edges:
                assert isinstance(e, mgp.Edge)
                updated.add(int(e.from_vertex.properties['id']))
            for e in v.out_edges:
                assert isinstance(e, mgp.Edge)
                updated.add(int(e.to_vertex.properties['id']))
            return mgp.Record(updated=updated)
    # 该节点没有邻居, 返回一个空集合
    return mgp.Record(updated=updated)


@mgp.write_proc
# 在算法最开始阶段，我们设置所有节点的度
# 为了简便，我们把所有层一起初始化
# ! 暂时弃用
def init_degree(context: mgp.ProcCtx):
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        v.properties['degree'] = len(list(v.in_edges) + list(v.out_edges))


# 获取某一层某一个节点的度
def get_degree(v: mgp.Vertex) -> mgp.Record(degree=mgp.Nullable[int]):
    return len(list(v.in_edges) + list(v.out_edges))


@mgp.read_proc
# 在最开始获取所有节点的度
def get_all_degree(context: mgp.ProcCtx, layer) -> mgp.Record(degree_list=mgp.Nullable[list]):
    degree_list = [0 for _ in range(V)]
    count = 0
    for v in context.graph.vertices:
        assert isinstance(v, mgp.Vertex)
        if int(v.properties['layer']) == layer:
            degree_list[int(v.properties['id'])] = get_degree(v)
            count += 1
    assert count == V
    return mgp.Record(degree_list=degree_list)
