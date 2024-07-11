# 测试mpyc的安全数组
from icecream import ic
from mpyc.runtime import mpc

sec_int = mpc.SecInt(13)


async def main():
    async with mpc:
        ls = [sec_int(1)]
        # 不能输入空数组
        ls = mpc.input(ls)
        ic(ls)
        ls = []
        ic(ls)
        # 可以输出空数组
        ls = await mpc.output(ls)

if __name__ == '__main__':
    mpc.run(main())