from mpyc.runtime import mpc
from icecream import ic

sec_int = mpc.SecInt(4)
sec_flt = mpc.SecFlt(5, 1)


async def main():
    async with mpc:
        s = sec_int(None)
        ic(s == 1)

if __name__ == '__main__':
    mpc.run(main())