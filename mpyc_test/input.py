from mpyc.runtime import mpc
from mpyc.seclists import seclist
from icecream import ic

SecInt4 = mpc.SecInt(4)

async def main():
    async with mpc:
        # ? 一直卡在这里(加了await)
        num_padding = mpc.sum([u_id == -1 for u_id in [1, 2, 4]])
        ic(num_padding)
        num_padding = await mpc.output(num_padding)
        ic(num_padding)

mpc.run(main())