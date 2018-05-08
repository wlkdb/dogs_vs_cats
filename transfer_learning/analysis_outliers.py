# -*- coding: utf-8 -*-
"""
Created on Mon May  7 23:31:33 2018

@author: liguo

异常数据分析
"""

from __future__ import print_function
from nets import nets_factory
from preprocessing import vgg_preprocessing
import sys
sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH

import tensorflow as tf
import os
import time
import shutil
import pandas as pd

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('num_classes', 2, 'The number of classes.')
tf.app.flags.DEFINE_string('infile', '../test', 'Image file, one image per line.')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to testuate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoint/','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('test_image_size', None, 'test image size.')
tf.app.flags.DEFINE_string('outliers_path', 'outliers', 'The path to save outliers images.')
FLAGS = tf.app.flags.FLAGS

model_name_to_variables = {'resnet_v1_50':'resnet_v1_50', 'vgg_16':'vgg_16'}

def main(_):
    model_variables = model_name_to_variables.get(FLAGS.model_name)
    if model_variables is None:
        tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
        sys.exit(-1)
    
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    
    # 读入图像、预处理模型、网络模型
    image_string = tf.placeholder(tf.string) 
    image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) 
    network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)
    
    # 数据预处理
    if FLAGS.test_image_size is None:
        test_image_size = network_fn.default_image_size
    processed_image = vgg_preprocessing.preprocess_image(image, test_image_size, test_image_size, is_training=False)
    processed_images    = tf.expand_dims(processed_image, 0) 
    
    # 获取输出
    logits, _ = network_fn(processed_images)
    probabilities = tf.nn.softmax(logits)
    
    # 初始化
    init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))
    sess = tf.Session()
    init_fn(sess)
    start_time = time.time()
    
    # 进行推断
    result = []
    test_images = os.listdir(FLAGS.infile)
    for test_image in test_images:
        path = os.path.join(FLAGS.infile, test_image)
        content = tf.gfile.FastGFile(path, 'rb').read()
        _logits, _prob = sess.run([logits, probabilities], feed_dict={image_string:content})
        sum_squares = _logits[0, 0] * _logits[0, 0] + _logits[0, 1] * _logits[0, 1]
        _prob = _prob[0, 0:]
        _prob = _prob[1]
        classes = 'cat' if 'cat' in test_image else 'dog'
        result.append([path, test_image, classes, sum_squares, _prob, _logits[0, 0], _logits[0, 1]])
			
    sess.close()
    
    # 将结果输出到csv文件
    path_list = []
    name_list = []
    class_list = []
    sum_squares_list = []
    prob_list = []
    logits1_list = []
    logits2_list = []
    for item in result:
        path_list.append(item[0])
        name_list.append(item[1])
        class_list.append(item[2])
        sum_squares_list.append(item[3])
        prob_list.append(item[4])
        logits1_list.append(item[5])
        logits2_list.append(item[6])
    dataframe = pd.DataFrame({'path':path_list, 'name':name_list, 'class':class_list, 
                              'sum_squares':sum_squares_list, 'prob':prob_list, 
                              'logits1':logits1_list, 'logits2':logits2_list})
    dataframe.to_csv("outliers.csv", index=False, sep=',')
    
    if not os.path.exists(FLAGS.outliers_path):
        os.makedirs(FLAGS.outliers_path)
    # 输出sum_squares最小的部分图片
    all_path = os.path.join(FLAGS.outliers_path, 'min_sum_squares')
    if not os.path.exists(all_path):
        os.makedirs(all_path)
    for i in range(min(500, len(result))):
        for j in range(i+1, len(result)):
            if result[i][3] > result[j][3]:
                temp = result[i]
                result[i] = result[j]
                result[j] = temp
        shutil.copyfile(result[i][0], os.path.join(all_path, format(i, "3d")+"_"+result[i][1]))
        
    # 输出cat中最难识别的部分图片
    cat_path = os.path.join(FLAGS.outliers_path, 'cat_max_logits')
    if not os.path.exists(cat_path):
        os.makedirs(cat_path)
    for i in range(min(250, len(result))):
        for j in range(i+1, len(result)):
            if (result[j][2] == 'cat') and (result[i][2] == 'dog' or result[i][4] < result[j][4]):
                temp = result[i]
                result[i] = result[j]
                result[j] = temp
        shutil.copyfile(result[i][0], os.path.join(cat_path, format(result[i][4], ".3f")+"_"+result[i][1]))
        
    # 输出dog中最难识别的部分图片
    dog_path = os.path.join(FLAGS.outliers_path, 'dog_min_logits')
    if not os.path.exists(dog_path):
        os.makedirs(dog_path)
    for i in range(min(250, len(result))):
        for j in range(i+1, len(result)):
            if (result[j][2] == 'dog') and (result[i][2] == 'cat' or result[i][4] > result[j][4]):
                temp = result[i]
                result[i] = result[j]
                result[j] = temp
        shutil.copyfile(result[i][0], os.path.join(dog_path, format(result[i][4], ".3f")+"_"+result[i][1]))
    
    print('total time cost = %.2f' %(time.time() - start_time))
    
if __name__ == '__main__':
    tf.app.run()
