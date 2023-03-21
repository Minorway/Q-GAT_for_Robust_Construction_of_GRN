import numpy as np
import matplotlib.pyplot as plt
import sys, copy, math, time, pdb,os
cur_dir = os.path.dirname(os.path.realpath(__file__))
from myfunctions import *

#利用随机数种子使每次生成的随机数相同
np.random.seed(10)
collectn_1 = np.random.normal(0, 1, (50,6))

path= 'E:\文件备份\GRGNN复件2.0(20220329）'

number = []                                    #创建一个列表
for i in range(0,50):                         #循环随机数100位
    num = random.randint(0,1)                  #num得到随机数
    number.append(num)                         #append是添加 随机数添加到number列表
print(number)

label=np.array(number)

print(collectn_1)
print(collectn_1.shape)

tsne_visual(collectn_1,label,cur_dir+'\VVVV')




