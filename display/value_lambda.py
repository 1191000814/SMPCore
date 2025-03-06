# 变化参数λ的值

# 随着计算方的增加
import paperplotlib as ppl
import numpy as np
from icecream import ic

# 比较算法1, 2, 3

# 数据集2的数据
y1 = [429.81, 420.29, 414.07, 410.35, 408.03, 405.67, 71.95, 27.29, 8.2, 3.47, 2.45, 2.27, 54.84, 22.27, 7.25, 3.39, 2.39, 2.26]
y2 = [88.46, 81.52, 82.1, 81.68, 76.84, 72.81, 45.34, 20.24, 12.78, 6.22, 4.21, 1.96, 35.43, 16.55, 11.12, 5.28, 4.01, 1.88]
y3 = [1440, 1440, 1440, 1440, 1440, 1440, 220.06, 219.14, 106.36, 106.99, 34.67, 35, 188.3, 188.5, 96.94, 97.29, 33.21, 33.61]
y4 = [1440, 1440, 1440, 1440, 1440, 1440, 102.11, 52.27, 35.78, 30.6, 25.83, 23.57, 91.98, 42.47, 29.5, 23.3, 19.57, 18.43]
y5 = [1.09, 0.914, 0.771, 0.704, 0.679, 0.492, 0.912, 0.579, 0.319, 0.17, 0.101, 0.04, 0.837, 0.52, 0.296, 0.154, 0.093, 0.039]
y6 = [1.78, 1.41, 1.35, 1.29, 1.28, 1.17, 1.48, 0.789, 0.571, 0.458, 0.354, 0.21, 1.44, 0.779, 0.603, 0.467, 0.375, 0.21]

# 多折线图
# 比较算法1, 2, 3
all_data = np.array([y1, y2, y3, y4, y5, y6])  # [10, 18]

line_names = ['SMPCore', 'SMPCore-BR', 'SMPCore-AR']

for i, y_data in enumerate(all_data):
    line_graph = ppl.LineGraph()
    line_graph.plt.rcParams['font.size'] = 12
    # line_graph.plt.gca().xaxis.set_label_position('right')
    x_data = [i + 1 for i in range(6)]
    line_graph.plot_2d(x_data, np.reshape(y_data, [3, 6]), line_names)
    line_graph.x_label = 'value of parameter λ'
    line_graph.y_label = 'running time (minute)'
    line_graph.plt.xlabel('value of parameter λ', fontdict={'fontsize': 6})
    line_graph.plt.ylabel('running time (minute)', fontdict={'fontsize': 6})
    # line_graph.plt.subplot(0.1, 0.9, 0.9, 0.1)
    # line_graph.plt.legend(fontsize=10)
    line_graph.plt.legend(ncol=2)
    line_graph.save(f'param_lambda{i}.png')