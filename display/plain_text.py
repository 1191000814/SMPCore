# 整体数据

import paperplotlib as ppl
import numpy as np
from icecream import ic

# y = [0.52, 0.42, 2.26, 0.286, 0.0185, 0.0336, 3.94, 4.45, 14.47, 0.17, 0.127, 0.252, 3.02, 3.35, 11.88, 0.169, 0.075, 0.192]
y = [0.52, 3.94, 3.02, 0.42, 4.45, 3.35, 2.26, 14.47, 11.88, 0.286, 0.17, 0.169, 0.0185, 0.127, 0.075, 0.0336, 0.252, 0.192]

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
bar_graph.x_label = "dataset"
bar_graph.y_label = "running time (s)"

# 保存图片
bar_graph.save('plaintext_result.png')
