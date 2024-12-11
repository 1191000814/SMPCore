# 整体数据

import paperplotlib as ppl
import numpy as np
from icecream import ic

y = [120, 20.99, 18.31, 27.72, 8.45, 7.94, 120, 56.16, 53.69, 120, 10.87, 10.41]
# 分组条形图数据
y_data = np.array(y).reshape([4, 3])
ic(y_data)

# 初始化一个对象
bar_graph = ppl.BarGraph()

# 传入数据/组/列的文字信息
group_names = ['homo', 'sacchcere', 'sanremo', 'slashdot']
column_names = ['SMPCore', 'SMPCore*', 'SMPCore-Switch']
bar_graph.plot_2d(y_data, group_names, column_names)

# 调整x/y轴文字
bar_graph.x_label = "Dataset"
bar_graph.y_label = "Running Time (Minute)"

# 保存图片
bar_graph.save('bar_graph.png')
