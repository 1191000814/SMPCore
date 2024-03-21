#!/bin/bash

# 计算方数量
n=3
m=$((n-1))
for i in `seq 0 $m`;
do
    python3 ~/Secure-Graph/no_db_demo/mpc_firmcore_deg.py -M$n -I$i > ../output/output_$i.txt 2>&1 &
    # python3 ~/Secure-Graph/no_db_demo/mpc_firmcore_mat.py -M$n -I$i > ../output/output_$i.txt 2>&1 &
done

# 等待所有后台进程完成（可选）
wait