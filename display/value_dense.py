# 变化密度的值

# 随着计算方的增加
import paperplotlib as ppl
import numpy as np
from icecream import ic

# 比较算法1,2,3

# 数据集2的数据
y2 = [4.08, 18.32, 27.72, 45.64, 65.67, 77.51, 1.69, 6.2, 8.45, 16.07, 34.16, 47, 1.7, 3.01, 7.94, 14.47, 28.63, 39.3]
# 合成数据集00
y00 = [0.16, 0.38, 0.83, 1.01, 1.87, 2.37, 0.072, 0.1, 0.16, 0.39, 0.36, 0.6, 0.067, 0.105, 0.183, 0.433, 0.398, 0.671]
# 合成数据集06
y06 = [19.4, 38.28, 66.75, 83.24, 110.18, 136.38, 4.93, 3.05, 3.5, 11.7, 6.6, 10, 4.76, 2.8, 3.59, 11.54, 6.61, 10.14]

# 比较算法1,2,3
all_data = np.array([y2, y06, y07, y08, y09, y10, y11, y12, y15, y16])  # [10, 18]

line_names = ['SMPCore', 'SMPCore*', 'SMPCore-Switch']

for i, y_data in enumerate(all_data):
    line_graph = ppl.LineGraph()
    x_data = [i + 1 for i in range(6)]
    line_graph.plot_2d(x_data, np.reshape(y_data, [3, 6]), line_names)
    line_graph.x_label = 'Number of Partys'
    line_graph.y_label = 'Running Time (Minute)'
    line_graph.save(f'line_graph{i}.png')
