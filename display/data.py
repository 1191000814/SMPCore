"""
main
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


datasets = ('homo', 'sacchcere', 'sanremo', 'slashdot')  # species
algs = ('alg1', 'alg2', 'alg3')  # attribute

times = {
    algs[0]: (10000, 1283, 10000, 10000),
    algs[1]: (808, 188, 2201, 461),
    algs[2]: (667, 172, 1874, 434),
}  # penguin_means

x = np.arange(len(datasets))  # the label locations
width = 0.25  # the width of the bars
internal = 0.05  # 同一组图里柱形的间隔
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for alg, time in times.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, time, width, label=alg)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Running Time (mintus)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, datasets)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 12000)

plt.show()

plt.savefig('./fig.png')
