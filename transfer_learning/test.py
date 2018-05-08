"""
@author: liguo

迁移学习的测试
"""

from __future__ import print_function
from nets import nets_factory
from preprocessing import vgg_preprocessing
import sys
sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH

import tensorflow as tf
import pandas as pd
import os
import time

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('num_classes', 2, 'The number of classes.')
tf.app.flags.DEFINE_string('infile', '../test', 'Image file, one image per line.')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to testuate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoint/','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('test_image_size', None, 'test image size.')

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
    
    result = []
    test_images = os.listdir(FLAGS.infile)
    for test_image in test_images:
        # 进行推断
        path = os.path.join(FLAGS.infile, test_image)
        content = tf.gfile.FastGFile(path, 'rb').read()
        probs = sess.run(probabilities, feed_dict={image_string:content})
        probs = probs[0, 0:]
        
        # 保存输出
        num = probs[1]
        border = 0.01
        if num > 1 - border:
            num = 1 - border
        elif num < border:
            num = border
        result.append([test_image[:-4], "%.3f"%num])
    sess.close()
    
    # 将结果按编号有小到大排序
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if int(result[i][0]) > int(result[j][0]):
                temp = result[i]
                result[i] = result[j]
                result[j] = temp
    
    # 将结果输出到csv文件
    id_list = []
    label_list = []
    for num in result:
        id_list.append(num[0])
        label_list.append(num[1])
    dataframe = pd.DataFrame({'id':id_list, 'label':label_list})
    dataframe.to_csv("result.csv", index=False, sep=',')
    
    print('total time cost = %.2f' %(time.time() - start_time))

if __name__ == '__main__':
    tf.app.run()