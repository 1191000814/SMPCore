import asyncio


async def fun1():
    print(1)


async def fun2():
    print(2)


async def main():
    # 创建任务和执行任务一定要在同一个函数内
    task1 = asyncio.create_task(fun1())
    task2 = asyncio.create_task(fun2())
    task1
    await task2


asyncio.run(main())
