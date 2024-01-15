# 测试mpyc的安全数组
from icecream import ic
from mpyc.runtime import mpc

sec_int = mpc.SecInt(4)


async def main():
    async with mpc:
        ls = [[sec_int(4), sec_int(2), sec_int(3)], [sec_int(4), sec_int(2), sec_int(3)]]
        ls = [mpc.input(a) for a in ls]
        ic(ls)

if __name__ == '__main__':
    mpc.run(main())
