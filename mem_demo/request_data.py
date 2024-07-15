import requests
from icecream import ic

dataset = 'TD'

res = requests.get(f'http://192.168.31.103:30089/public/multiplex_dataset/FirmTruss/{dataset}.txt')
ic(res.status_code)

file_name = f"{dataset}.txt"

# 打开文件用于写入
with open(file_name, 'w', encoding='utf-8') as file:
    # 将字符串写入文件
    file.write(res.text)
