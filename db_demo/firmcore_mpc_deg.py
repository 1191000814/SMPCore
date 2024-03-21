# 每一方都在自己本地运行该代码, 只能从input中获取其他方的数据(加密)
# 只使用自己数据计算时, 无需加密
# 将数据传给其他方, 或者获取其他方的数据时, 都需要加密

import numpy as np
import mgp_mock as mgp
import firmcore_mage
from neo4j import GraphDatabase
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
from test_demo.nx_graph_test import create_3layer_1

URI = "bolt://localhost:7687"
AUTH = ('', '')

# 自己预先设置图的节点数和层数
V = 15
L = 3
PADDING = -1  # 填充用的数字, 不和v_id重合, 且不超过最大数字范围

secint4 = mpc.SecInt(16)


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


async def firmcore(lamb):
    # 处于安全我们考虑添加一个受信任的第三方, 其余每一层各作为安全计算的一方
    with GraphDatabase.driver(URI, auth=AUTH) as client:
        # 初始化所有的度
        ic(client.verify_authentication())
        client.execute_query('CALL file_core.init_degree()')
        async with mpc:
            # 每个顶点vertices对应的Top-λ(deg(vertices))
            I = [None for _ in range(V)]
            # B[i]表示Top-λ(deg(vertices))为i的顶点有哪些, 因为度的最大值只能为VV
            B = [set() for _ in range(V - 1)]
            # 查询所有的度
            records, _, _ = client.execute_query(
                f'CALL s_file_core.get_all_deg({mpc.pid}, {i})')
            # 该层所有的度, list[V * int]
            deg_list = records[0]['deg_list']
            assert isinstance(deg_list, list)
            # 换成安全类型, np.array(V, )
            sec_degree_array = np.array([secint(d) for d in deg_list])
            # 从mpc.input到mpc.output之间的数据都是加密的,output之后就没有加密了
            # Degree指的是所有的度, np.array(V * L)
            Degree = mpc.input(sec_degree_array)
            Degree = np.array(Degree)
            # 在加密状态下求每个v第λ大的度
            for i in range(V):
                # id为i的顶点在各层的第λ大的度
                top_d = mpc.sorted(Degree[:, i])[-lamb]
                # ic(top_d)
                I[i] = top_d
                B[I[i]].add(i)
                assert I[i] <= V
            I1 = await mpc.output(I)
            ic(I1)
            core = [set() for _ in range(V - 1)]
            # 下面是根据k逐渐增大来依次淘汰顶点
            for k in range(V - 1):
                while B[k]:
                    ic(I)
                    ic(B)
                    # 在每一层分别中去掉这个节点
                    v_id = B[k].pop()
                    ic(v_id)
                    core[k].add(v_id)
                    N = set()
                    for l in range(l):
                        record = client.execute_query(
                            f'CALL file_core.remove_node({mpc.pid}, {i})')
                        update_nodes = records[0]['update_nodes']
                        assert isinstance(update_nodes, dict)
                        for u_id, d in update_nodes.items():
                            if d == I[u_id] - 1 and I[u_id] > k:
                                B[I[u_id]].remove(I[u_id])
                                top_d = mpc.sorted(
                                    Degree[:, u_id])[-lamb]  # 第lamb+1大的度
                                I[u_id] = top_d
                                B[I[u_id]].add(u_id)
            core = await mpc.output(core)
            ic(core)


# 上面函数的mock测试版本
# 同时使用数据结构I和B
async def firmcore_mock_degree1(lamb):
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        firmcore_mage.init_degree(context)
        I = [V for _ in range(V)]
        # B为一个VV个元素的数组 I[i]为一个二元数组p
        # p[1]为SecInt(i), p[2]为所有Top-λ(deg(v))为i的节点id
        B = [[None, []] for _ in range(V - 1)]
        # 该层所有的度, list[V * int]
        deg_list = firmcore_mage.get_all_deg(
            context, mpc.pid).fields['deg_list']
        assert isinstance(deg_list, list)
        # 换成安全类型, np.array(V, )
        sec_deg_list = [secint(d) for d in deg_list]
        # 从mpc.input到mpc.output之间的数据都是加密的,output之后就没有加密了
        # Degree指的是所有的度, np.array(V * L)
        Degree = mpc.input(sec_deg_list)
        Degree = np.array(Degree)
        # 在加密状态下求每个v第λ大的度
        for i in range(V - 1):
            # id为i的顶点在各层的第λ大的度
            top_degree = mpc.sorted(Degree[:, i])[-lamb]
            I[i] = top_degree
            # B[I[i]].add(i) # 原版本
            for p in B:
                # p为二元组(安全整数: i, 安全整数集合: 第λ大的度为i的所有顶点id)
                if p[0] is None:
                    assert not p[1]
                    p[0] = top_degree
                    p[1].append(secint(i))
                    break
                # elif p[0] == top_degree:
                #     assert p[1]
                #     p[1].add(i)
                #     break
                else:
                    # 因为p[0]不为None, 那么p[1]也不为空数组
                    p[1] = mpc.if_else(p[0] == top_degree,
                                       p[1] + [secint(i)], p[1])
            # assert I[i] <= secint(V)
        # I = await mpc.output(I)
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(V)] for _ in range(V - 1)]
        # 下面是根据k逐渐增大来依次淘汰顶点
        v_list = []
        for k in range(V - 1):
            for p in B:
                v_list = mpc.if_else(p[0] == secint(k), p[1], v_list)
                # if p[0] == secint(k):
                #     v_set = p[1]
            while v_list:
                ic(I)
                ic(B)
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回修改过的节点和度, 再次input
                record = firmcore_mage.remove_node(context, {mpc.pid}, {i})
                updated = record.fields['updated']
                assert isinstance(updated, dict)
                # 修改后度为I[id]的节点
                secure_updated = []
                # 有修改的度
                if updated:
                    for u_id, d in updated.items():
                        if d == I[u_id] - 1 and I[u_id] > k:
                            secure_updated.append(secint(u_id))
                # 不同的层之间传的节点可能有重复
                secure_updated = mpc.input(secure_updated)
                for u_id in secure_updated:
                    for p in B:
                        if p[0] == secint(u_id):
                            p[1].remove(I[u_id])
                            top_degree = mpc.sorted(
                                Degree[:, u_id])[-lamb]  # 第lamb+1大的度
                            I[u_id] = top_degree
                            p[1].add(u_id)
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firmcore_mock_degree2_0(lamb):
    # 修改版本, 删去数据结构B, 仅仅使用I
    # 前面的-1都保留下来, 等到最后再处理, 除了选择v_list时, 其他地方避免使用明文
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        deg_list = firmcore_mage.get_all_deg(
            context, mpc.pid).fields['deg_list']
        assert isinstance(deg_list, list)
        # 换成安全类型的list
        secure_deg_list = [secint(d)
                           for d in deg_list] + [secint(PADDING)]
        # * 从mpc.input输出的数据都是加密的了, 因为获取了其他方的数据
        Degree = np.array(mpc.input(secure_deg_list))
        # Degree的[行]为层数, [列]为id
        Degree = [seclist(per_layer, secint4) for per_layer in Degree]
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度
            per_id = [Degree[l][i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        # output时必须是list形式
        ic(await mpc.output(list(I)))
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(PADDING)] for _ in range(V - 1)]
        ic('下面是根据k逐渐增大来依次淘汰顶点')
        for k in range(V - 1):
            ic(f'---------{k}---------')
            v_list = seclist([mpc.if_else(I[v_id] == k, v_id, PADDING)
                             for v_id in range(V)], secint4)
            ic(await mpc.output(list(v_list)))
            n = await mpc.output(v_list.count(PADDING))
            ic(n)
            # 得到的v_list前l位不为0, 后面全为0, 截取前面不为0的位
            v_list.sort()
            del v_list[:n]
            v_list_out = await mpc.output(list(v_list))
            ic(v_list_out)
            v_list = list(v_list)
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                ic(await mpc.output(v_id))
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回v的邻居节点id, 再次input
                v_id0 = await mpc.output(v_id)
                updated = firmcore_mage.remove_node(
                    context, mpc.pid, v_id0).fields['updated']
                ic(updated)
                assert isinstance(updated, list)
                # 所有需要修改degree和I[v_id]的节点id
                secure_updated = [mpc.if_else(I[u_id] > k, u_id, V)
                                  for u_id in updated]
                # 如果数组为空, 在后面的count函数中会报错
                secure_updated.append(PADDING)
                # 需要删除的元素
                # ? 一直卡在这里(加了await)
                ic(secure_updated)
                secure_updated.append(PADDING)
                # seclist不能作为input, 转化为[secint...]
                # 各个层中修改过度的节点
                secure_updated = mpc.input(secure_updated)
                ic(secure_updated)
                # 所有需要修改I值的节点id(不区分层)
                u_all_layer = seclist([], secint4)
                for l, u_per_layer in enumerate(secure_updated):
                    # 每一个需要修改I的u_id
                    for u_id in u_per_layer:
                        # 首先需要先更新Degree矩阵
                        # u_id0 = await mpc.output(u_id)
                        u_id0 = await mpc.output(secint(10))
                        Degree[l, mpc.if_else(
                            u_id0 != PADDING, u_id0, 0)] = mpc.if_else(
                            u_id0 != PADDING, Degree[l, u_id0] - 1, Degree[l, 0])
                        # * 不同的层之间传的节点可能有重复
                        u_all_layer.append(mpc.if_else(u_id0 != PADDING and not u_all_layer.contains(u_id)
                                                       and Degree[l, u_id0] == I[u_id] - 1, u_id, PADDING))
                ic(u_all_layer)
                if u_all_layer:
                    # n = await mpc.output(u_all_layer.count(PADDING))
                    # u_all_layer.sort()
                    # 删除前面n个值为V的节点id
                    # del u_all_layer[V - n:]
                    # ic(u_all_layer)
                    # 更新I
                    for u_id in u_all_layer:
                        top_d = mpc.sorted(
                            Degree[:, await mpc.output(u_id)])[-lamb]  # 更新top-degree
                        I[mpc.if_else(u_id != PADDING, u_id, PADDING)] = top_d
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firmcore_mock_degree2_1(lamb):
    # 上面函数的mock测试版本
    # 修改版本, 删去数据结构B, 仅仅使用I
    # ? 修改每次output都使用一次mpc.start()和shutdown()
    G = create_3layer_1()
    context = mgp.ProcCtx(G)
    ic(f'{mpc.pid + 1} of {L} party')
    # 获取整个多层图的度矩阵
    deg_list = firmcore_mage.get_all_deg(
        context, mpc.pid).fields['deg_list']
    assert isinstance(deg_list, list)
    # 换成安全类型的list
    secure_deg_list = [secint(d)
                       for d in deg_list] + [secint(PADDING)]
    # * 从mpc.input输出的数据都是加密的了, 因为获取了其他方的数据
    # Degree指的是所有的度, 形状为 (V + 1) * L, 最后一个数字用于填充
    await mpc.start()
    Degree = np.array(mpc.input(secure_deg_list))
    await mpc.shutdown()
    # 每个顶点vertices对应的Top-λ(deg(vertices))
    ic('创建数组I')
    I = seclist([-1 for _ in range(V)], secint4)
    ic('初始化数组I为每个顶点第λ大的度')
    for i in range(V):
        # id为i的顶点在各层的第λ大的度
        I[i] = mpc.sorted(Degree[:, i])[-lamb]
    # output时必须是list形式
    ic(await mpc.output(list(I), receivers=mpc.pid))
    # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
    core = [[secint(PADDING)] for _ in range(V - 1)]
    ic('下面是根据k逐渐增大来依次淘汰顶点')
    for k in range(V - 1):
        ic(f'---------{k}---------')
        v_list = seclist([mpc.if_else(I[v_id] == k, v_id, PADDING)
                          for v_id in range(V)], secint4)
        ic(await mpc.output(list(v_list)))
        n = await mpc.output(v_list.count(PADDING))
        ic(n)
        # 得到的v_list前l位不为0, 后面全为0, 截取前面不为0的位
        v_list.sort()
        del v_list[:n]
        v_list_out = await mpc.output(list(v_list))
        ic(v_list_out)
        v_list = list(v_list)
        await mpc.shutdown()
        while v_list:
            # 在每一层分别中去掉这个节点
            v_id = v_list.pop(0)
            ic(await mpc.output(v_id))
            core[k].append(v_id)
            # 每一方更新自己层的顶点的度, 返回v的邻居节点id, 再次input
            v_id0 = await mpc.output(v_id)
            updated = firmcore_mage.remove_node(
                context, mpc.pid, v_id0).fields['updated']
            ic(updated)
            assert isinstance(updated, list)
            # 所有需要修改degree和I[v_id]的节点id
            secure_updated = seclist([mpc.if_else(I[u_id] > k, u_id, V)
                                      for u_id in updated], secint4)
            # 如果数组为空, 在后面的count函数中会报错
            secure_updated.append(PADDING)
            # 需要删除的元素
            # ? 一直卡在这里(加了await)
            # n = await mpc.output(secint(3))
            ic(secure_updated)
            # 不能使mpc.sum求和的数组为空, 至少填充一个元素
            secure_updated.append(PADDING)
            # seclist不能作为input, 转化为[secint...]
            # 各个层中修改过度的节点
            await mpc.start()
            secure_updated = mpc.input(list(secure_updated))
            await mpc.shutdown()
            ic(secure_updated)
            # 所有需要修改I值的节点id(不区分层)
            u_all_layer = seclist([], secint4)
            for l, u_per_layer in enumerate(secure_updated):
                # 每一个需要修改I的u_id
                for u_id in u_per_layer:
                    # 首先需要先更新Degree矩阵
                    u_id0 = await mpc.output(u_id)
                    Degree[l, mpc.if_else(
                        u_id0 != PADDING, u_id0, 0)] = mpc.if_else(
                        u_id0 != PADDING, Degree[l, u_id0] - 1, Degree[l, 0])
                    # * 不同的层之间传的节点可能有重复
                    u_all_layer.append(mpc.if_else(u_id0 != PADDING and not u_all_layer.contains(u_id)
                                                   and Degree[l, u_id0] == I[u_id] - 1, u_id, PADDING))
            ic(u_all_layer)
            if u_all_layer:
                for u_id in u_all_layer:
                    top_d = mpc.sorted(
                        Degree[:, await mpc.output(u_id)])[-lamb]  # 更新top-degree
                    I[mpc.if_else(u_id != PADDING, u_id, PADDING)] = top_d
        core = [await mpc.output(c) for c in core]
        ic(core)


async def firmcore_mock_degree2_2(lamb):
    # 上面函数的mock测试版本
    # 修改版本, 删去数据结构B, 仅仅使用I
    # 使用mpc.is_zero_public, 这样可以使用if关键字来判断了
    # 修改Degree矩阵结构为[seclist, seclist, ...], 每个seclist为每个id在各层的节点
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        ic(f'{mpc.pid + 1} of {L} party')
        # 获取整个多层图的度矩阵
        deg_list = firmcore_mage.get_all_deg(
            context, mpc.pid).fields['deg_list']
        assert isinstance(deg_list, list)
        # 换成安全类型的list
        secure_deg_list = [secint(d)
                           for d in deg_list] + [secint(PADDING)]
        # * 从mpc.input输出的数据都是加密的了, 因为获取了其他方的数据
        # Degree指的是所有的度, 形状为 (V + 1) * L, 最后一个数字用于填充
        Degree = np.array(mpc.input(secure_deg_list))
        # Degree的[行]为层数, [列]为id
        Degree = [seclist(per_layer, secint4) for per_layer in Degree]
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic('创建数组I')
        I = seclist([-1 for _ in range(V)], secint4)
        ic('初始化数组I为每个顶点第λ大的度')
        for i in range(V):
            # 每个id在各层的度
            per_id = [Degree[l][i] for l in range(L)]
            # id为i的顶点在各层的第λ大的度
            I[i] = mpc.sorted(per_id)[-lamb]
        # output时必须是list形式
        ic(await mpc.output(list(I)))
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint(PADDING)] for _ in range(V - 1)]
        ic('下面是根据k逐渐增大来依次淘汰顶点')
        for k in range(V - 1):
            ic(f'---------{k}---------')
            v_list = []
            for v_id in range(V):
                if await mpc.eq_public(I[v_id], k):
                    v_list.append(v_id)
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                # ic(await mpc.output(v_id))
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回v的邻居节点id, 再次input
                # v_id0 = await mpc.output(v_id)
                ic('删除这次选中的节点, 并修改其邻居的度')
                updated = firmcore_mage.remove_node(
                    context, mpc.pid, v_id).fields['updated']
                ic(updated)
                assert isinstance(updated, list)
                # 所有需要修改degree和I[v_id]的节点id
                secure_updated = []
                for u_id in updated:
                    if await mpc.eq_public((I[u_id] > k), True):
                        secure_updated.append(secint(u_id))
                # 输入的数组不为空
                if not secure_updated:
                    secure_updated.append(secint(PADDING))
                secure_updated = mpc.input(secure_updated)
                u_all_layer = seclist([], secint4)
                ic('遍历每层修改过度的节点')
                for l, u_per_layer in enumerate(secure_updated):
                    # 每一个需要修改I的u_id
                    for u_id in u_per_layer:
                        # 首先需要先更新Degree矩阵
                        # u_id0 = await mpc.output(u_id)
                        if not mpc.eq_public(u_id, -1):
                            Degree[l][u_id] -= 1
                            # * 不同的层之间传的节点可能有重复
                            if await mpc.eq_public(Degree[l][u_id], I[u_id] - 1):
                                u_all_layer.append(u_id)
                ic(u_all_layer)
                if u_all_layer:
                    for u_id in u_all_layer:
                        per_id = [Degree[l][u_id] for l in range(L)]
                        # id为i的顶点在各层的第λ大的度
                        I[u_id] = mpc.sorted(per_id)[-lamb]
        core = [await mpc.output(c) for c in core]
        ic(core)

if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(firmcore_mock_degree2_2(2))
