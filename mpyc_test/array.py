# 测试mpyc的安全数组
from icecream import ic
from mpyc.runtime import mpc

sec_int = mpc.SecInt(4)


async def main():
    async with mpc:
        ls1 = [sec_int(1), sec_int(2), sec_int(3)]
        ls = mpc.input(ls1[:mpc.pid + 1])
        ic(ls)

if __name__ == '__main__':
    mpc.run(main())
