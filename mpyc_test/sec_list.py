from mpyc.seclists import seclist, secindex
from mpyc.runtime import mpc
from icecream import ic

secint = mpc.SecInt(4)
secfxt = mpc.SecFlt(4)


async def main():
    async with mpc:
        ls = seclist([i for i in range(6)], sectype=secint)
        ls = mpc.input(ls)
        ic(type(ls))
        ic(len(ls))
        ic(ls[2:])


mpc.run(main())
