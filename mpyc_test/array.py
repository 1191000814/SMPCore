# 测试mpyc的安全数组
from icecream import ic
from mpyc.runtime import mpc
from mpyc.sectypes import SecureInteger

sec_int = mpc.SecInt(4)


async def main():
    async with mpc:
        t = [1, 2, 3]
        u = mpc.seclist([-1] * (len(t)+1), sec_int)
        for i in range(len(t), 0, -1):
            u[mpc.sum(t[:i])] = i - 1
        b = list(u[1:])

if __name__ == '__main__':
    mpc.run(main())
    