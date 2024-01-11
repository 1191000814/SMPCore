import asyncio


def gen(n):
    for i in range(n):
        print(f'gen {i}')
        yield i


async def fun1():
    # async表示异步, 不能直接调用, 只能用asyncio.run()或者其他异步包调用
    # 也可以被其他异步函数调用
    print(1)


async def fun2():
    # await表示等待协程(加上了async的函数)fun1执行完再执行
    await fun1()
    print(2)

# asyncio.create_task([fun1(), fun2()])
asyncio.run(fun1())
asyncio.run(fun2())

it = gen(10)
for i in it:
    print(i)
