from mpyc.runtime import mpc
from mpyc.sectypes import SecureInteger
from icecream import ic

secint4 = mpc.SecInt(16)


def secint(num):
    # 返回加密的4-bit数字
    return secint4(num)


class SecureNode:
    # 用来在邻接表中存储邻居的链表节点
    def __init__(self, value):
        # ic(value)
        if isinstance(value, int):
            value = secint(value)
        self.value = value
        self.next = None

    def __add__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value + other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value + other.value)
        else:
            raise TypeError('unsupport type for add')

    def __sub__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value - other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value - other.value)
        else:
            raise TypeError('unsupport type for sub')

    def __mul__(self, other):
        if isinstance(other, SecureInteger):
            return SecureNode(self.value * other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value * other.value)
        else:
            raise TypeError('unsupport type for mul')

    def __rmul__(self, other):
        ic(111)
        if isinstance(other, SecureInteger):
            return SecureNode(self.value * other)
        elif isinstance(other, SecureNode):
            return SecureNode(self.value * other.value)
        else:
            raise TypeError('unsupport type for rmul')


async def main():
    async with mpc:
        a = secint(2) * SecureNode(1)
        ic(a)


mpc.run(main())