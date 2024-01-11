import mgp_mock as mgp
# import mgp


@mgp.read_proc
# 找到某个城市所在的国家(节点)
def find_city_in(context: mgp.ProcCtx, find_city: mgp.Nullable[str]) -> mgp.Record(country=mgp.Nullable[str]):
    for city in context.graph.vertices:  # 遍历所有的节点
        assert (isinstance(city, mgp.Vertex))
        # label = city.labels[0]
        # return mgp.Record(country=label.name)
        if city.properties['name'] == find_city:
            for r in city.out_edges:
                if r.type == 'Inside':
                    return mgp.Record(country=r.to_vertex.properties['name'])
    return mgp.Record(country=None)


@mgp.write_proc
# 写过程: 添加一个在某个国家中的新城市
def new_city(context: mgp.ProcCtx,
             in_city: mgp.Nullable[str],  # 添加的city
             in_country: mgp.Nullable[str]  # 添加的country
             ) -> mgp.Record(is_finished=bool):
    for country in context.graph.vertices:  # 遍历图中所有的点
        assert (isinstance(country, mgp.Vertex))
        # 找到了指定的Country
        if (country.labels[0] == 'Country') & (country.properties['name'] == in_country):
            city = context.graph.create_vertex()  # 创建该城市
            city.add_label('City')  # 添加标签和属性
            city.properties.set('name', in_city)
            city.properties.set('country', in_country)
            # 与所属国家的边"Inside"
            context.graph.create_edge(
                city, country, mgp.EdgeType('Inside'))
            return mgp.Record(is_finished=True)
    # 没有指定的country
    return mgp.Record(is_finished=False)  # 不加参数名就会报错


@mgp.read_proc
def list() -> mgp.Record(list=mgp.Nullable[list]):
    return mgp.Record(list=[1, 2, 3])