#coding=utf-8
"""
@author: liguo

从头训练的测试
"""

import tensorflow as tf
import data_preprocess
import resnet
import math
import time
import pandas as pd 
    
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('checkpoint_path', "checkpoint/", 'The path of checkpoint dir.')
tf.app.flags.DEFINE_integer('net_mode', 0, 'The deep of resnet.')
FLAGS = tf.app.flags.FLAGS

N_CLASSES = 2 #猫和狗
IMG_W = 224 #输入图像尺寸
IMG_H = 224 
BATCH_SIZE = 64 
CAPACITY = 2000

test_dir = '../test/' #测试集位置
checkpoint_dir = FLAGS.checkpoint_path #checkpoint保存位置

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
    
# 获取输入数据
test, test_label = data_preprocess.get_test_files(test_dir)
test_batch, test_label_batch=data_preprocess.get_batch(test,
                                test_label,
                                IMG_W,
                                IMG_H,
                                BATCH_SIZE,
                                CAPACITY,
                                is_training=False)
# 将输入命名以便在Android app中使用
test_batch = tf.add(test_batch, tf.zeros([IMG_W, IMG_H, 3]), name="input")
one_hot_labels = tf.one_hot(indices=tf.cast(test_label_batch, tf.int32), depth=2)

# 获取输出
learning_rate = tf.placeholder(tf.float32)
train_op, train_loss, train_logits = resnet.inference(test_batch, one_hot_labels, 1500, \
                                                                 deep=N_CLASSES, is_training=False, batch_size=BATCH_SIZE, lr=learning_rate, \
                                                                 net_mode=FLAGS.net_mode)
correct_prediction = tf.equal(tf.argmax(train_logits, 1), tf.argmax(one_hot_labels, 1))
train__acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
# 初始化
sess = tf.Session() 
coord = tf.train.Coordinator() #队列监控
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
total_begin_time = time.time()

# checkpoint读取
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and ckpt.model_checkpoint_path:
    print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

# 进行测试
length = len(test)
result = []
for i in range(int(math.ceil(length/BATCH_SIZE)) + 2):
    if coord.should_stop():
        break            
    probabilities = tf.nn.softmax(train_logits)
    indexs, probs = sess.run([test_label_batch, probabilities])
    for j in range(len(indexs)):
        num = probs[j, 1]
        border = 0.01
        if num > 1 - border:
            num = 1 - border
        elif num < border:
            num = border
        result.append([int(indexs[j]), "%.3f"%num])

# 将测试结果按编号有小到大排序
for i in range(len(result)):
    for j in range(i+1, len(result)):
        if int(result[i][0]) > int(result[j][0]):
            temp = result[i]
            result[i] = result[j]
            result[j] = temp
            
# 将测试结果输出为.csv文件
id_list = []
label_list = []
last_id = -1
for num in result:
    if num[0] != last_id: #去掉重复的编号
        id_list.append(num[0])
        label_list.append(num[1])
    last_id = num[0]
dataframe = pd.DataFrame({'id':id_list, 'label':label_list})
dataframe.to_csv("output.csv", index=False, sep=',')
    
print('total time cost = %.2f' %(time.time() - total_begin_time))
    
coord.request_stop()
    
    
    
