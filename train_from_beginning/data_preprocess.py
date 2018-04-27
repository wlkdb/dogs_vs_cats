#coding=utf-8
"""
@author: liguo

数据预处理
"""

import tensorflow as tf
import numpy as np
import os
import random

# 获取乱序后的文件路径和标签
def get_files(file_dir):
    cats = []
    label_cats = []
    dogs = []
    label_dogs = []
    # 载入数据路径并写入标签值
    for file in os.listdir(file_dir):
        if file.find("cat") >= 0: 
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)
    print("There are %d cats\nThere are %d dogs" % (len(cats), len(dogs)))

    # 打乱文件顺序
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    temp = np.array([image_list, label_list])
    temp = temp.transpose()     
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list

# 获取文件路径和去掉后缀名后的文件名
def get_test_files(file_dir):
    image_list = []
    label_list = []
    for file in os.listdir(file_dir):
        image_list.append(file_dir + file)
        label_list.append(int(file[:-4]))
    return image_list, label_list

# 随机覆盖，覆盖区域置黑
def random_cover(image):
    mat = np.ones(image.get_shape())
    #覆盖区域的大小占总图片大小的比例
    SCALE = 5 
    height = image.get_shape().as_list()[0]
    width = image.get_shape().as_list()[1]
    begin_H = random.randint(0, height - int(height / SCALE) - 1)
    begin_W = random.randint(0, width - int(width / SCALE) - 1)
    mat[begin_H:begin_H+int(height/SCALE), begin_W:begin_W+int(width/SCALE), :] = 0
    mat_tensor = tf.convert_to_tensor(mat, tf.uint8)
    return tf.multiply(image, mat_tensor)
    
# 对image进行数据预处理，并返回batch_size大小的Tensor
def get_batch(image, label, image_W, image_H, batch_size, capacity, is_training=True, mode=9):
    # image, label: 要生成batch的图像和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量

    # 将python.list类型转换成tf能够识别的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # 生成队列
    input_queue = tf.train.slice_input_producer([image, label])
    image_contents = tf.read_file(input_queue[0])
    label = input_queue[1]
    image = tf.image.decode_jpeg(image_contents, channels=3)
    
    if not is_training:
        image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.cast(image, tf.float32)
    else:

        # 初始处理
        if mode == 0:
            image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        
        # 随机裁剪
        if mode > 1:
            T = 5 if mode == 2 else 4 #裁剪比例
            image = tf.image.resize_images(image, [int(image_H*T/(T-1)), int(image_W*T/(T-1))], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            image = tf.random_crop(image, [image_H, image_W, 3])
        
        image = tf.cast(image, tf.float32)
        
        # 随机翻转
        if mode == 4 or mode == 8:
            image = tf.image.random_flip_left_right(image)
        
        # 随机覆盖
        if mode == 5:
            image = random_cover(image)
        
        # 随机亮度与对比度
        if mode == 6 or mode == 8:
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
        
        # 随机饱和度与色调
        if mode == 7:
            image = tf.image.random_saturation(image, lower=0.2, upper=1.8)
            image = tf.image.random_hue(image, max_delta=0.5)
        
    # 图片标准化
    if mode == 1:
        image = tf.image.per_image_standardization(image)   
    
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=64,   
                                              capacity=capacity)
    return image_batch, label_batch
