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

    def __add__(self, other):
        return SecureNode(self.value + other.value)

    def __sub__(self, other):
        return SecureNode(self.value - other.value)

    def __mul__(self, other):
        return SecureNode(self.value * other.value)

    def __rmul__(self, other):
        return SecureNode(self.value * other)


async def main():
    async with mpc:
        I_copy = seclist([1, 2, 3], secint)
        while I_copy:
            a = I_copy.pop()
            print(a)

mpc.run(main())
