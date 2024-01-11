from mpyc.runtime import mpc
from icecream import ic

SecInt4 = mpc.SecInt(4)

async def main():
    async with mpc:
        a = SecInt4(1)
        a = mpc.input(a)
        a = await mpc.output(a)
        ic(a)
        b = SecInt4(2)
        b = mpc.input(b)
        b = await mpc.output(b)
        ic(b)

mpc.run(main())