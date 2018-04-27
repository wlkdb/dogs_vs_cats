#coding=utf-8
"""
@author: liguo

从头训练的验证
"""

import numpy as np
import tensorflow as tf
import data_preprocess
import resnet
import math
import time
    
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('checkpoint_path', "checkpoint/", 'The path of checkpoint dir.')
tf.app.flags.DEFINE_integer('net_mode', 0, 'The deep of resnet.')
FLAGS = tf.app.flags.FLAGS

N_CLASSES = 2 #猫和狗
IMG_W = 224 #输入图像尺寸
IMG_H = 224 
BATCH_SIZE = 64 
CAPACITY = 2000

verify_dir = 'verify/' #验证集位置
checkpoint_dir = FLAGS.checkpoint_path #checkpoint保存位置

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
    
# 获取输入数据
verify, verify_label = data_preprocess.get_files(verify_dir)
verify_batch, verify_label_batch=data_preprocess.get_batch(verify,
                                verify_label,
                                IMG_W,
                                IMG_H,
                                BATCH_SIZE,
                                CAPACITY,
                                is_training=False)
# 将输入命名以便在Android app中使用
verify_batch = tf.add(verify_batch, tf.zeros([IMG_W, IMG_H, 3]), name="input")
one_hot_labels = tf.one_hot(indices=tf.cast(verify_label_batch, tf.int32), depth=2)

# 获取输出
learning_rate = tf.placeholder(tf.float32)
train_op, train_loss, train_logits = resnet.inference(verify_batch, one_hot_labels, 1500, \
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
    
# 进行验证
length = len(verify)
loss_list = []
acc_list = []
for i in range(int(math.ceil(length/BATCH_SIZE))):
    if coord.should_stop():
        break
    tra_loss, tra_acc = sess.run([train_loss, train__acc])
    loss_list.append(tra_loss)
    acc_list.append(tra_acc)
result_loss = np.mean(loss_list)
result_acc = np.mean(acc_list)

print("loss = %.4f  acc = %.2f" % (result_loss, result_acc*100.0))                
print('total time cost = %.2f' %(time.time() - total_begin_time))

coord.request_stop()
    