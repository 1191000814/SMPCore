from mpyc.seclists import seclist, secindex
from mpyc.runtime import mpc
from icecream import ic
import mpyc.sectypes as stype

secint = mpc.SecInt(16)
secfxt = mpc.SecFlt(4)


class SecureNode:
    # 用来在邻接表中存储邻居的链表节点
    def __init__(self, value):
        # ic(value)
        self.value = value
        self.next = None


class A:
    def __init__(self, a) -> None:
        self.a = a


async def main():
    async with mpc:
        # 取值方法, 取从offset开始的
        I = seclist([secint(1), secint(2)], secint)
        i = secindex([secint(1)], offset=1)
        a = I[i]
        ic(await mpc.output(a))

mpc.run(main())
