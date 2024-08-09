#!/bin/bash

cd /home/xiaozeqiang/Secure-Graph/mem_demo
pwd
# 运行一个数据集在不同算法, 不同计算方的多个结果
M=6
d='00'
v=("v1-1" "v2-1" "v3-1")

for ((m=2; m<=M; m++)); do # 计算方数量为m
    for ((i=0; i<m; i++)); do # 第i方的计算程序
        python3 mpc_firmcore.py -M$m -d$d -v${v[0]} -I$i > ../output/output_M${m}_m${i}_d${d}_v${v[0]}.txt 2>&1 &
    done
    # sleep 1
    for ((i=0; i<m; i++)); do
        python3 mpc_firmcore.py -M$m -d$d -v${v[1]} -I$i > ../output/output_M${m}_m${i}_d${d}_v${v[1]}.txt 2>&1 &
    done
    # sleep 1
    for ((i=0; i<m; i++)); do
        python3 mpc_firmcore.py -M$m -d$d -v${v[2]} -s3 -I$i > ../output/output_M${m}_m${i}_d${d}_v${v[2]}_s3.txt 2>&1 &
    done
    # sleep 1
done