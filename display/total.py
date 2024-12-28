# 整体数据

import paperplotlib as ppl
import numpy as np
from icecream import ic

y = [184.7, 16.28, 13.65, 23.06, 8.88, 7.47, 1440, 37.33, 31.93, 1440, 7.54, 7.14, 0.039, 0.0253, 0.023, 0.0476, 0.0322, 0.0297]
# 分组条形图数据
y_data = np.array(y).reshape([6, 3])
ic(y_data)

# 初始化一个对象
bar_graph = ppl.BarGraph()

# 传入数据/组/列的文字信息
group_names = ['Homo', 'Sacchcere', 'Sanremo', 'Slashdot', 'Terrorist', 'RM']
column_names = ['SMPCore', 'SMPCore-BP', 'SMPCore-AP']
bar_graph.plot_2d(y_data, group_names, column_names, log=True)

# 调整x/y轴文字
bar_graph.x_label = "Dataset"
bar_graph.y_label = "Running Time (Minute)"

# 保存图片
bar_graph.save('bar_graph.png')
