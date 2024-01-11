# 处理数据集并导入到memgraph数据库
# 数据结构:
# 多层、无向、无权重图, 每一层代表一个航空公司, 每个节点是一个欧洲的机场
# 每条边代表该航空公司在两个机场之间有无直飞航线
# 数据格式见README
# 生成图是无向图, 但是memgraph只有有向图, 那么每两点之间要么没有边, 要么有不同方向的两条边
# 所有的id, 有序数均为0开头

from neo4j import GraphDatabase
from icecream import ic

V = 450
L = 37        

create_all = 'CREATE'
with open('./data/air-multi-public-dataset/airports.txt', 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        values = line.split()
        # 每行的数值分别是机场id, 所属国际组织, 经度, 纬度
        # 第l层的节点被创建为n{l * V + id}
        assert len(values) == 4
        for layer in range(L):
            # 创建时节点变量n__后面的数字就是节点的id
            create_all += f"(n{layer * V + idx}:Airport{{ID:'{int(values[0]) - 1}', \
                layer:'{layer}', ICAO:'{values[1]}', longitude:'{values[2]}', latitude:'{values[3]}'}}),"

# create_all = create_all[:-1] + ';\nCRETAE'  # 去掉最后一个逗号

# 创建边和节点用一个CREATE关键字就行, 否则会报错
with open('./data/air-multi-public-dataset/network.txt', 'r', encoding='utf-8') as file:
    layer = -1  # 层数
    node_idx = 0  # 遍历该行的顶点
    node_num = 0  # 该顶点的邻居数(从数据中获取)
    edge_num = 0  # 该顶点的邻居数(计数得出)
    total_edge_num = 0  # 所有的边数
    for idx, line in enumerate(file):
        values = line.split()
        if len(values) == 1:  # 表示这一层的顶点数, 也表示上一个层已经结束
            ic(layer, node_idx, node_num)
            assert node_idx == node_num
            node_num = int(values[0])
            node_idx = 0
            layer += 1
        elif len(values) >= 3:
            node_id = values[0]  # 创建边的节点
            edge_num = int(values[1])  # 创建边的数量
            assert edge_num == len(values) - 2
            for n_id in values[2:]:
                create_all += f'(n{layer * V + int(node_id)})-[r{total_edge_num}:Route]->(n{layer * V + int(n_id)}),'
                edge_num += 1
                total_edge_num += 1
            node_idx += 1
    assert layer == L - 1
    ic(total_edge_num)
    create_all = create_all[:-1] + ';'  # 去掉最后一个逗号

URI = 'bolt://localhost:7687'
AUTH = ('', '')

print(create_all)

with GraphDatabase.driver(URI, auth=AUTH) as client:
    ic(client.verify_authentication())
    client.execute_query('MATCH (n) DETACH DELETE n')
    client.execute_query(create_all)
