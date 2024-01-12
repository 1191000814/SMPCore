from mpyc.seclists import seclist, secindex
from mpyc.runtime import mpc
from icecream import ic

secint = mpc.SecInt(4)
secfxt = mpc.SecFlt(4)


async def main():
    async with mpc:
        ls = seclist([], sectype=secint)
        n = await mpc.output(ls.count(-1))
        ic(n)
mpc.run(main())
