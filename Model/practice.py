import numpy as np
import matplotlib.pyplot as plt
# 利用 numpy库生成三组正态分布随机数
x = [np.random.normal(0, std, 100) for std in range(1, 4)]
print('x',x)
# 绘图
plt.boxplot(x,
            showfliers=False,
            patch_artist=True, sym='o',
            labels=['A', 'B', 'C'],  # 添加具体的标签名称
            showmeans=False,
            boxprops={'color': 'black', 'facecolor': '#9999ff'}, #'#9999ff'  #'White'
            flierprops={'marker': 'o', 'markerfacecolor': 'red', 'color': 'black'},
            meanprops={'marker': 'D', 'markerfacecolor': 'indianred', 'color': 'y', },
            medianprops={'linestyle': '--', 'color': 'orange'})

# 显示图形

plt.show()