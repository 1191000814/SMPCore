# 测试mpyc的安全数组
from icecream import ic
from mpyc.runtime import mpc

sec_int = mpc.SecInt(4)


async def main():
    async with mpc:
        pass

if __name__ == '__main__':
    mpc.run(main())