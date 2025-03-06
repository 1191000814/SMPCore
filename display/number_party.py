# 随着计算方的增加
import paperplotlib as ppl
import numpy as np
from icecream import ic

# 比较算法1,2,3
# [算法1的6个数据, 算法2的6个数据, 算法3的6个数据]

y1 = [49.64, 128.91, 197.28, 269.88, 354.86, 485.65, 6.44, 16.13, 18.4, 22.39, 54.05, 75.76, 5.65, 13.39, 15.19, 17.44, 47.69, 59.32]
y2 = [4.08, 18.32, 27.72, 45.64, 65.67, 77.51, 7.29, 9.39, 18.99, 34.88, 44.67, 45.95, 5.41, 7.53, 15.38, 27.58, 34.97, 35.04]
y3 = [686.6, 1087, None, None, None, None, 38.69, 47.89, 34.19, 53.97, 57.37, 68.81, 30.75, 42.44, 30.74, 41.14, 50.07, 60.88]
y4 = [488.53, 927.6, None, None, None, None, 1.79, 4.12, 4.58, 7.8, 8.63, 11.28, 1.72, 3.81, 4.27, 7.52, 8.27, 11.27]

# 多折线图
# 比较算法1,2,3
all_data = np.array([y1, y2, y3, y4])  # [4, 18]
line_names = ['SMPCore', 'SMPCore-BR', 'SMPCore-AR']
group_names = ['Homo', 'Sacchcere', 'Sanremo', 'Slashdot', 'Terrorist', 'RM']

for i, y_data in enumerate(all_data):
    line_graph = ppl.LineGraph()
    line_graph.plt.rcParams['font.size'] = 16
    x_data = [i + 1 for i in range(6)]
    line_graph.plot_2d(x_data, np.reshape(y_data, [3, 6]), line_names)
    # line_graph.plt.title(group_names[i])
    line_graph.x_label = '|L|'
    line_graph.y_label = 'running time (min)'
    line_graph.plt.xlabel('value of parameter λ', fontdict={'fontsize': 8})
    line_graph.plt.ylabel('running time (minute)', fontdict={'fontsize': 8})
    line_graph.plt.legend(fontsize=8)
    line_graph.save(f'number_of_party{i}.png')