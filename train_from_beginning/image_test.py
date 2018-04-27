# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 05:03:40 2018

@author: liguo

将数据预处理的图像结果输出到文件
"""
import tensorflow as tf
import numpy as np
import random

# 随机覆盖，覆盖区域置黑
def random_cover(image):
    mat = np.ones(image.get_shape())
    
    SCALE = 5
    height = image.get_shape().as_list()[0]
    width = image.get_shape().as_list()[1]
    begin_H = random.randint(0, height - int(height / SCALE) - 1)
    begin_W = random.randint(0, width - int(width / SCALE) - 1)
    mat[begin_H:(begin_H+int(height/SCALE)), begin_W:(begin_W+int(width/SCALE)), :] = 0

    mat_tensor = tf.convert_to_tensor(mat, tf.uint8)
    print(image)
    print(mat_tensor)
    return tf.multiply(image, mat_tensor)

image_contents = tf.read_file("../train/dog.10708.jpg")
image = tf.image.decode_jpeg(image_contents, channels=3)

image_W = 224
image_H = 224

# 初始处理
#image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

# 随机裁剪
T = 4 #裁剪比例
image = tf.image.resize_images(image, [int(image_H*T/(T-1)), int(image_W*T/(T-1))], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
image = tf.random_crop(image, [image_H, image_W, 3])

image = tf.cast(image, tf.float32)

# 随机覆盖
#image = random_cover(image)
        
# 随机翻转
image = tf.image.random_flip_left_right(image)
        
# 随机亮度与对比度
image = tf.image.random_brightness(image, max_delta=0.5)
image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        
# 随机饱和度与色调
#image = tf.image.random_saturation(image, lower=0.2, upper=1.8)
#image = tf.image.random_hue(image, max_delta=0.5)

# 图片标准化
#image = tf.image.per_image_standardization(image)   
    
# 保存图片到文件
with tf.Session() as sess:
    img_data = tf.image.convert_image_dtype(image, dtype= tf.uint8)
    encoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.GFile('data_progress_output.jpg', 'wb') as f:
        f.write(encoded_image.eval())    