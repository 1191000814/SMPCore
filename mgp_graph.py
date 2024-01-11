# 自己创建数据集
import queue
import re
from neo4j import GraphDatabase
from icecream import ic

V = 13  # 顶点数
L = 3  # 层数
edges = [set() for _ in range(L)]
# 在每一层加入边(一条边就是一个二元组)
edges[0].update(
    {(1, 2), (1, 3), (1, 8), (2, 3), (3, 8), (4, 5), (4, 6), (5, 6), (5, 7), (5, 9), (5, 10), (6, 7), (9, 10), (9, 11),
     (9, 13), (10, 11), (10, 13), (11, 12), (11, 13)})
edges[1].update(
    {(2, 3), (2, 4), (2, 7), (3, 4), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 6), (5, 7), (7, 8), (9, 10),
     (9, 11), (9, 12), (10, 12), (11, 12), (12, 13)})
edges[2].update(
    {(1, 2), (1, 3), (1, 8), (2, 3), (2, 7), (2, 8), (3, 4), (3, 7), (3, 8), (4, 5), (4, 6), (4, 7), (4, 8), (5, 6),
     (5, 7), (6, 7), (7, 8), (9, 10), (9, 11), (9, 12), (9, 13), (11, 12), (11, 13), (12, 13)})
E = sum(len(layer) for layer in edges)  # 总共的边数
ic(E)

# Define correct URI and AUTH arguments (no AUTH by default)
URI = "bolt://localhost:7687"
AUTH = ('', '')

with GraphDatabase.driver(URI, auth=AUTH) as client:
    assert client.verify_authentication()
    client.verify_connectivity()
    # 创建之前先删除原有节点
    delete_all = ' MATCH(n) DETACH DELETE(n);'
    client.execute_query(delete_all)
    # 创建节点
    for l in range(L):
        # 创建节点的语句
        create_node = "CREATE (:Node {id: $id, layer: $layer});"
        # 创建边的语句
        create_rel = "MATCH (v1:Node{id: $id1, layer: $layer}),(v2:Node{id: $id2, layer: $layer})\
            CREATE (v1)-[r:REl]->(v2) RETURN r"
        for i in range(V):
            # 创建节点
            client.execute_query(create_node, id=str(i), layer=str(str(l)))
        for e in edges[l]:
            # 创建边
            client.execute_query(create_rel, id1=str(e[0] - 1), id2=str(e[1] - 1), layer=str(l))
    num_node, _, _ = client.execute_query(
        'MATCH(n) RETURN count(n) AS num_node;')
    num_node = num_node[0]['num_node']
    ic(num_node)
    assert num_node == V * L
    num_ref, _, _ = client.execute_query(
        'MATCH()-[r]->() RETURN count(r) AS num_rel;')
    num_ref = num_ref[0]['num_rel']
    ic(num_ref)
    assert num_ref == E
