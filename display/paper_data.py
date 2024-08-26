import paperplotlib as ppl
import numpy as np
from icecream import ic

y = [120, 20.99, 18.31, 27.72, 8.45, 7.94, 120, 56.16, 53.69, 120, 10.87, 10.41]
# 分组条形图数据
y_data = np.array(y).reshape([4, 3])
ic(y_data)

# 初始化一个对象
graph = ppl.BarGraph()

# 传入数据/组/列的文字信息
group_names = ['homo', 'sacchcere', 'sanremo', 'slashdot']
column_names = ['alg1', 'alg2', 'alg3']
graph.plot_2d(y_data, group_names, column_names)

# 调整x/y轴文字
graph.x_label = "Dataset"
graph.y_label = "Running Time (Minute)"

# 保存图片
graph.save('bar_graph.png')

# 比较算法1,2,3

# 数据集2的数据
y2 = [4.08, 18.32, 27.72, 45.64, 65.67, 77.51, 1.69, 6.2, 8.45, 16.07, 34.16, 47, 1.7, 3.01, 7.94, 14.47, 28.63, 39.3]
# 合成数据集00
y00 = [0.16, 0.38, 0.83, 1.01, 1.87, 2.37, 0.072, 0.1, 0.16, 0.39, 0.36, 0.6, 0.067, 0.105, 0.183, 0.433, 0.398, 0.671]
# 合成数据集06
y06 = [19.4, 38.28, 66.75, 83.24, 110.18, 136.38, 4.93, 3.05, 3.5, 11.7, 6.6, 10, 4.76, 2.8, 3.59, 11.54, 6.61, 10.14]
# 多折线图

# 比较算法1,2,3
y_data = np.array(y00).reshape([3, 6])
# 仅仅比较算法2,3
y_data_23 = np.array(y06[int(len(y2) / 3) :]).reshape([2, 6])

ic(y_data)

line_names = ['alg1', 'alg2', 'alg3']

graph = ppl.LineGraph()
x_data = [i + 1 for i in range(6)]
graph.plot_2d(x_data, y_data_23, line_names)
graph.x_label = 'Number of Partys'
graph.y_label = 'Running Time (Minute)'
graph.save('line_graph.png')
