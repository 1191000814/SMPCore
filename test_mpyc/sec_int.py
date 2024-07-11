from mpyc.runtime import mpc
from icecream import ic

sec_int = mpc.SecInt(16)
sec_flt = mpc.SecFlt(5, 1)


async def main():
    async with mpc:
        b = mpc.max(sec_int(1), 2)
        ic(await mpc.output(b))

if __name__ == '__main__':
    mpc.run(main())
