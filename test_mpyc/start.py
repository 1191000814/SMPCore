from mpyc.runtime import mpc
from icecream import ic

sec_int = mpc.SecInt(4)


async def main():
    await mpc.start()
    a = mpc.input(sec_int(1))
    await mpc.shutdown()
    ic(a)
    await mpc.start()
    a = mpc.input(sec_int(1))
    await mpc.shutdown()
    ic(await mpc.output(a))

mpc.run(main())
