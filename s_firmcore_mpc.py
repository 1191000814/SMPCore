# 每一方都在自己本地运行该代码, 只能从input中获取其他方的数据(加密)
# 只使用自己数据计算时, 无需加密
# 将数据传给其他方, 或者获取其他方的数据时, 都需要加密

import numpy as np
import mgp_mock as mgp
import s_firmcore_mage
from neo4j import GraphDatabase
from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic
from nx_graph import create_3layer_1

URI = "bolt://localhost:7687"
AUTH = ('', '')

# 自己预先设置图的节点数和层数
V = 15
L = 3


def secint4(num):
    # 返回加密的4-bit数字
    secint4 = mpc.SecInt(4)
    return secint4(num)


async def secure_firm_core(lamb):
    # 处于安全我们考虑添加一个受信任的第三方, 其余每一层各作为安全计算的一方
    with GraphDatabase.driver(URI, auth=AUTH) as client:
        # 初始化所有的度
        ic(client.verify_authentication())
        client.execute_query('CALL file_core.init_degree()')
        async with mpc:
            # 每个顶点vertices对应的Top-λ(deg(vertices))
            I = [None for _ in range(V)]
            # B[i]表示Top-λ(deg(vertices))为i的顶点有哪些, 因为度的最大值只能为V-1
            B = [set() for _ in range(V - 1)]
            # 查询所有的度
            records, _, _ = client.execute_query(
                f'CALL s_file_core.get_all_degree({mpc.pid}, {i})')
            # 该层所有的度, list[V * int]
            degree_list = records[0]['degree_list']
            assert isinstance(degree_list, list)
            # 换成安全类型, np.array(V, )
            sec_degree_array = np.array([secint4(d) for d in degree_list])
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
async def secure_firm_core_mock(lamb):
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        s_firmcore_mage.init_degree(context)
        I = [-1 for _ in range(V)]
        # B为一个V-1个元素的数组 I[i]为一个二元数组p
        # p[1]为SecInt(i), p[2]为所有Top-λ(deg(v))为i的节点id
        B = [[None, []] for _ in range(V - 1)]
        # 该层所有的度, list[V * int]
        degree_list = s_firmcore_mage.get_all_degree(
            context, mpc.pid).fields['degree_list']
        assert isinstance(degree_list, list)
        # 换成安全类型, np.array(V, )
        sec_degree_list = [secint4(d) for d in degree_list]
        # 从mpc.input到mpc.output之间的数据都是加密的,output之后就没有加密了
        # Degree指的是所有的度, np.array(V * L)
        Degree = mpc.input(sec_degree_list)
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
                    p[1].append(secint4(i))
                    break
                # elif p[0] == top_degree:
                #     assert p[1]
                #     p[1].add(i)
                #     break
                else:
                    # 因为p[0]不为None, 那么p[1]也不为空数组
                    p[1] = mpc.if_else(p[0] == top_degree,
                                       p[1] + [secint4(i)], p[1])
            # assert I[i] <= secint4(V)
        # I = await mpc.output(I)
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint4(-1)] for _ in range(V - 1)]
        # 下面是根据k逐渐增大来依次淘汰顶点
        v_list = []
        for k in range(V - 1):
            for p in B:
                v_list = mpc.if_else(p[0] == secint4(k), p[1], v_list)
                # if p[0] == secint4(k):
                #     v_set = p[1]
            while v_list:
                ic(I)
                ic(B)
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回修改过的节点和度, 再次input
                record = s_firmcore_mage.remove_node(context, {mpc.pid}, {i})
                updated = record.fields['updated']
                assert isinstance(updated, dict)
                # 修改后度为I[id]的节点
                secure_updated = []
                # 有修改的度
                if updated:
                    for u_id, d in updated.items():
                        if d == I[u_id] - 1 and I[u_id] > k:
                            secure_updated.append(secint4(u_id))
                # 不同的层之间传的节点可能有重复
                secure_updated = mpc.input(secure_updated)
                for u_id in secure_updated:
                    for p in B:
                        if p[0] == secint4(u_id):
                            p[1].remove(I[u_id])
                            top_degree = mpc.sorted(
                                Degree[:, u_id])[-lamb]  # 第lamb+1大的度
                            I[u_id] = top_degree
                            p[1].add(u_id)
        core = [await mpc.output(c) for c in core]
        ic(core)


async def secure_firm_core_mock2(lamb):
    # 上面函数的mock测试版本
    # 修改版本, 删去数据结构B, 仅仅使用I
    G = create_3layer_1()
    context = mgp.ProcCtx(G)

    async with mpc:
        # 每个顶点vertices对应的Top-λ(deg(vertices))
        ic(f'{mpc.pid + 1} of {L} party')
        s_firmcore_mage.init_degree(context)
        I = [-1 for _ in range(V)]
        degree_list = s_firmcore_mage.get_all_degree(
            context, mpc.pid).fields['degree_list']
        assert isinstance(degree_list, list)
        # 换成安全类型, np.array(V, )
        sec_degree_list = [secint4(d) for d in degree_list]
        # 从mpc.input到mpc.output之间的数据都是加密的,output之后就没有加密了
        # Degree指的是所有的度, np.array(V * L)
        Degree = mpc.input(sec_degree_list)
        Degree = np.array(Degree)
        # 在加密状态下求每个v第λ大的度
        for i in range(V):
            # id为i的顶点在各层的第λ大的度
            top_degree = mpc.sorted(Degree[:, i])[-lamb]
            # ic(top_degree)
            I[i] = top_degree
        # I = await mpc.output(I)
        # core是最终要返回的结果, 但是也需要加密, 否则将不能输出
        core = [[secint4(-1)] for _ in range(V - 1)]
        # 下面是根据k逐渐增大来依次淘汰顶点
        for k in range(V - 1):
            ic(f'---------{k}---------')
            # is_k: 哪些加密元素和k相同
            is_k = seclist([k == top_d for top_d in I])
            num_k = await mpc.output(mpc.sum(list(is_k)))
            ic(num_k)
            # v_list: 哪些元素的top-d值等于k, 该数组的长度最后肯定会被得知, 所以直接设置成明文
            # 最后一个位置为填充位
            v_list = seclist([-1 for _ in range(num_k + 1)],
                             sectype=mpc.SecInt(4))
            for i in range(len(is_k)):
                # 如果该元素是k, 则将其写入列表下一个位置, 否则写到最后的填充位
                # 遍历到i有几个True
                count_k = mpc.sum(list(is_k[:i + 1]))
                # 下个要赋值的索引
                next_ix = mpc.if_else(is_k[i] == 1, count_k - 1, V)
                v_list[next_ix] = secint4(i)
            # 得到的v_list前l位不为0, 后面全为0, 截取前面不为0的位
            v_list = v_list[:num_k]
            v_list_out = await mpc.output(list(v_list))
            ic(v_list_out)
            while v_list:
                # 在每一层分别中去掉这个节点
                v_id = v_list.pop(0)
                ic(await mpc.output(v_id))
                core[k].append(v_id)
                # 每一方更新自己层的顶点的度, 返回修改过的节点和度, 再次input
                updated = s_firmcore_mage.remove_node(
                    context, mpc.pid, await mpc.output(v_id)).fields['updated']
                assert isinstance(updated, dict)
                # 所有需要修改I[id]的度, 每一层input为[节点id, 度, 节点id, 度]
                secure_updated = []
                if updated:
                    # 不同的层之间传的节点可能有重复
                    for u_id, degree in updated.items():
                        if I[u_id] > k:
                            secure_updated += [secint4(u_id), secint4(degree)]
                # 将空数组传入mpc.input可能会报错, 所以至少传入一个元素
                ic(updated)
                if not secure_updated:
                    secure_updated.append(secint4(-1))
                # 各层更新了度的节点, 和更新之后的度
                secure_updated = mpc.input(secure_updated)
                # 所有需要修改I值的节点id(和层无关)
                u_all_layer = []
                for l, u_per_layer in enumerate(secure_updated):
                    # 每一个需要修改I的u_id
                    if len(u_per_layer) >= 2:
                        assert len(u_per_layer % 2) == 0
                        for i in range(len(u_per_layer) / 2):
                            # 首先需要先更新Degree矩阵
                            u_id, degree = u_per_layer[i], u_per_layer[i + 1]
                            Degree[l, u_id] = degree
                            u_all_layer = mpc.if_else(
                                degree == I[u_id] - 1, u_all_layer + [secint4(u_id)], u_all_layer)
                # 更新I
                for u_id in u_all_layer:
                    for i in range(V - 1):
                        I[i] = mpc.if_else(u_id == secint4(i), mpc.sorted(
                            Degree[:, u_id])[-lamb], I[i])  # 更新top-degree
        core = [await mpc.output(c) for c in core]
        ic(core)

if __name__ == '__main__':
    # mpc.run(secure_firm_core())
    mpc.run(secure_firm_core_mock2(2))
