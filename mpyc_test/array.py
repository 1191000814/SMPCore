# 测试mpyc的安全数组
from icecream import ic
from mpyc.runtime import mpc

sec_int = mpc.SecInt(4)


async def main():
    async with mpc:
        ls1 = [sec_int(1), sec_int(2)]
        ls2 = [sec_int(3), sec_int(4)]
        mpc.np_fromlist()
        ls = mpc.if_else(sec_int(1) > 0, ls1, ls2)
        ls.append(sec_int(3))
        ic(ls1)
        ic(ls2)


if __name__ == '__main__':
    mpc.run(main())
