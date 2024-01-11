from neo4j import GraphDatabase
from icecream import ic

# Define correct URI and AUTH arguments (no AUTH by default)
URI = "bolt://localhost:7687"
AUTH = ('', '')


with GraphDatabase.driver(URI, auth=AUTH) as client:
    # Check the connection
    client.verify_connectivity()
    ic(client.supports_multi_db())
    # Find a user John in the database
    records, summary, keys = client.execute_query(
        "MATCH (c:City) RETURN c.name AS city_name, c.country AS country_name",
    )

    for record in records:
        ic(record['city_name'], record['country_name'])
    for key in keys:
        ic(keys)
    ic(summary.counters)


# 打开dirver，连接图数据库
with GraphDatabase.driver(URI, auth=AUTH) as client:
    # 检测是否连接成功
    client.verify_connectivity()

    # 执行一条cypher语句，相当于在memgraph-lab的编辑框中执行
    # 返回的三个值分别是
    # --records: 查询结果记录
    # --summary: 查询结果总结
    # --keys: 查询结果中包括哪些关键字, 这里为city_name
    records, summary, keys = client.execute_query(
        "MATCH (c:City {country: $country}) RETURN c.name AS city_name;",
        country='China'
    )

    # records是查询的结果数组
    for record in records:
        print(record["name"])

    # keys是查询的
    for key in keys:
        print(key)
    # summary是该查询的聚合操作, 例如counter是查询出的条目的数量
    print(summary.counters)
    # query是查询的语句
    print(summary.query)