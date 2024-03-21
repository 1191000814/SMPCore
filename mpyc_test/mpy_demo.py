# 每一个party输入一个年龄,经过安全计算之后输出:
# 1.最大年龄(public)
# 2.年龄比自己大的还有几个人(private)

from mpyc.runtime import mpc
from icecream import ic

sec_int = mpc.SecInt(2)

async def main():
    async with mpc:
        x = mpc.input(sec_int(mpc.pid))
        y = await mpc.output(mpc.sum(x))
        if mpc.pid == 0:
            print(f'Hello 0, result:{y}')
        elif mpc.pid == 1:
            print(f'Hello 1, result:{y}')
        else:
            print('Sorry, no result')
    async with mpc:
        x = mpc.input(sec_int(int(input('Enter:'))))
        # x表示多方的输入的x组成的一个加密对象的list
        x = mpc.np_fromlist(x)
        y = await mpc.output(mpc.np_sum(x))
        if mpc.pid == 0:
            print(f'Hello 0, result:{y}')
        elif mpc.pid == 1:
            print(f'Hello 1, result:{y}')
        else:
            print('Sorry, no result')

if __name__ == '__main__':
    mpc.run(main())
