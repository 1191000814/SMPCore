from mpyc.seclists import seclist, secindex
from mpyc.runtime import mpc
from icecream import ic

secint = mpc.SecInt(16)
secfxt = mpc.SecFlt(4)

P = 9999

async def main():
    async with mpc:
        ls = seclist([], sectype=list)
        ls.append([secint(1), secint(2)])
        ls.append([1, 2])
        ic(ls[-1])
        ic(ls[0])

mpc.run(main())
