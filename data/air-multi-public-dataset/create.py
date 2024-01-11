# 处理数据集并导入到memgraph数据库
# 数据结构:
# 多层、无向、无权重图, 每一层代表一个航空公司, 每个节点是一个欧洲的机场
# 每条边代表该航空公司在两个机场之间有无直飞航线
# 数据格式见README

from neo4j import GraphDatabase
from icecream import ic

LINE_NUM = 450


with open('./airports.txt', 'r', encoding='utf-8') as file:
    create_all_nodes = 'CREATE'
    for idx, line in enumerate(file):
        values = line.split()
        # 每行的数值分别是机场id, 所属国际组织, 经度, 纬度
        assert len(values) == 4
        create_all_nodes += f"(n{idx + 1}:Airport{{ID:'{values[0]}', ICAO:'{values[1]}',\
        longitude:'{values[2]}', latitude:'{values[3]}'}})"
        if idx < LINE_NUM - 1:
            create_all_nodes += ','


URI = 'bolt://localhost:7687'
AUTH = ('', '')

print(create_all_nodes)

with GraphDatabase.driver(URI, auth=AUTH) as client:
    ic(client.verify_authentication())
    client.execute_query('MATCH (n) DETACH DELETE n')
    client.execute_query(create_all_nodes)
