from mpyc.runtime import mpc
from icecream import ic

sec_int = mpc.SecInt(4)


async def main():
    ic(mpc.parties)
    ic(len(mpc.parties))

mpc.run(main())
