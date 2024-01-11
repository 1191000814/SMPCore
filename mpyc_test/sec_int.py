from mpyc.runtime import mpc
from icecream import ic

sec_int = mpc.SecInt(4)
sec_flt = mpc.SecFlt(5, 1)


async def main():
    async with mpc:
        a = sec_int(1)
        b = sec_int(2)
        c1 = mpc.if_else(a == 2, a, b)
        ic(c1 == 1)
        c2 = mpc.if_else(a == 1, a, b)
        ic(c1, c2)


if __name__ == '__main__':
    mpc.run(main())
