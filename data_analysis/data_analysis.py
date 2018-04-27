# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 06:16:53 2018

@author: liguo

绘制数据集图片尺寸分布热力图，输出图片数量较多的尺寸区间
"""
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import tensorflow as tf
from matplotlib.ticker import LogFormatter 
import os

tf.app.flags.DEFINE_string('path', "../train", 'The path of data set.')
FLAGS = tf.app.flags.FLAGS

PATH = FLAGS.path

# 坐标轴绘制范围
SIZE = 1200
# 坐标轴缩放系数
T = 10

arr = np.zeros((int(SIZE / T), int(SIZE / T)))

# 图片尺寸统计
test_images = os.listdir(PATH)
for test_image in test_images:
    path = os.path.join(PATH, test_image)
    img = Image.open(path)
    width = int(img.width / T)
    height = int(img.height / T)
    arr[width, height] += 1
    
for i in range(int(SIZE/T)):
    for j in range(int(SIZE/T)):
        if arr[i][j] > 100:
            print(i, ' ', j, ' ', arr[i][j])
    
nums = [0, 20, 40, 60, 80, 100]
labels = [0] + [x*T for x in nums]
fig = plt.figure(1, figsize=(7, 7))   
ax = fig.add_subplot(1, 1, 1)       
ax.set_yticks(nums)       
ax.set_yticklabels(labels)       
ax.set_xticks(nums)      
ax.set_xticklabels(labels)      
ax.set_xlabel("width")
ax.set_ylabel("height")
ax.set_title(FLAGS.path[2:] + " set")
mapp = ax.matshow(arr, cmap='hot', norm=colors.LogNorm())
formatter = LogFormatter(10, labelOnlyBase=False) 
cb = plt.colorbar(mappable=mapp, ticks=[1, 10, 100, 1000], format=formatter)

plt.show()