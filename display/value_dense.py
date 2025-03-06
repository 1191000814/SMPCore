# 变化密度的值

# 随着计算方的增加
import paperplotlib as ppl
import numpy as np
from icecream import ic

# 比较算法1,2,3

# 数据集2的数据
y1 = [4.08, 18.32, 27.72, 45.64, 65.67, 77.51, 1.69, 6.2, 8.45, 16.07, 34.16, 47, 1.7, 3.01, 7.94, 14.47, 28.63, 39.3]


# 比较算法1,2,3
all_data = np.array([y2, y06, y07, y08, y09, y10, y11, y12, y15, y16])  # [10, 18]

line_names = ['SMPCore', 'SMPCore-BR', 'SMPCore-AR']

for i, y_data in enumerate(all_data):
    line_graph = ppl.LineGraph()
    x_data = [i + 1 for i in range(6)]
    line_graph.plot_2d(x_data, np.reshape(y_data, [3, 6]), line_names)
    line_graph.x_label = 'Number of Partys'
    line_graph.y_label = 'Running Time (Minute)'
    line_graph.save(f'line_graph{i}.png')
