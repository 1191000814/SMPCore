# 测试python中的运算符重载
from icecream import ic


class A:

    def __init__(self, a) -> None:
        self.a = a

    def __add__(self, other):
        return A(self.a + other)

    def __rsub__(self, other):
        return A(self.a - other)

    def __radd__(self, other):
        return A(self.a + other)


a = A(1)
b = 1 - a
ic(b.a)
