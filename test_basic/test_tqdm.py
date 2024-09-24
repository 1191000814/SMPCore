from tqdm import tqdm
from time import sleep
from icecream import ic

# for i in tqdm(range(100)):
#     sleep(0.1)

with tqdm(total=100) as pbar:
    for i in range(100):
        pbar.update(2)
        sleep(0.1)
        # ic(i)
