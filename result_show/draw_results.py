# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:15:58 2018

@author: liguo

绘制各模型kaggle得分柱状图
"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def draw(labels, results):
    width = 0.4
    ind = np.linspace(0.5, 9.5, 8)
    fig = plt.figure(1, figsize=(11, 7))
    ax  = fig.add_subplot(111)
    ax.bar(ind-width/2, results, width, color='green')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.set_ylabel('loss')
    plt.grid(True)
    plt.show()
    plt.close()
    
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] 

labels = ["初始模型", '数据增强', '模型深度', '优化器', '学习速率', '全训练集', '迁移学习', '迁移学习(全训练集)']

results = [0.90861, 0.69662, 0.60714, 0.37441, 0.30311, 0.11477, 0.06669, 0.05504]

draw(labels, results)





