from icecream import ic
import numpy as np
from mpyc.runtime import mpc
from mpyc.seclists import seclist

secint = mpc.SecInt(16)


class SecureNode:
    # 用来在邻接表中存储邻居的链表节点
    def __init__(self, value):
        # ic(value)
        if isinstance(value, int):
            value = secint(value)
        self.value = value
        self.next = None

    def append(self, value):
        # 在链表最后添加一个元素
        if isinstance(value, int):
            value = secint(value)
        p = self
        while p.next is not None:
            p = p.next
        p.next = SecureNode(value)


ic(1, end=',')
