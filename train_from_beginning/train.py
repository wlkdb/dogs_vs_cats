#coding=utf-8
"""
@author: liguo

从头训练的训练
"""

import os
import numpy as np
import tensorflow as tf
import data_preprocess
import resnet
import math
import sys
is_py2 = sys.version[0] == '2'
if is_py2:
    import Queue
else:
    import queue as Queue
    
import time
from tensorflow.python.framework import graph_util
    
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', "train_s/", 'The path of train data.')
tf.app.flags.DEFINE_string('checkpoint_path', "checkpoint/", 'The path of checkpoint dir.')
tf.app.flags.DEFINE_integer('data_preprocess', 8, 'The mode of data preprocess.')
tf.app.flags.DEFINE_integer('net_mode', 0, 'The deep of resnet.')
tf.app.flags.DEFINE_integer('opt_mode', 1, 'The type of optimizer.')
tf.app.flags.DEFINE_integer('lr_mode', 1, 'The type of learning rate.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'The size of a batch.')
FLAGS = tf.app.flags.FLAGS

N_CLASSES = 2 #猫和狗
IMG_W = 224 #输入图像尺寸
IMG_H = 224 
BATCH_SIZE = FLAGS.batch_size 
CAPACITY = 2000
MAX_LOOP = 300 #训练最大迭代次数
EARLY_STOP_NUM = 10 #触发早期停止的迭代次数
if FLAGS.data_preprocess > 0: #采用数据增强时
    EARLY_STOP_NUM += 5
if FLAGS.lr_mode == 5: #采用周期性余弦衰减学习速率时
    EARLY_STOP_NUM += 10

train_dir = FLAGS.train_dir #训练集位置
checkpoint_dir = FLAGS.checkpoint_path #checkpoint保存位置
pb_file_path = 'catsAndDogs.pb' #pb文件保存位置
    
# 获取输入数据
train, train_label = data_preprocess.get_files(train_dir)
DATA_SIZE = len(train) #训练集大小
MAX_STEP = int(DATA_SIZE / BATCH_SIZE * MAX_LOOP + 1) #训练最大步数
SAVE_INTERVAL = int(DATA_SIZE / BATCH_SIZE * 10) #checkpoint保存检查间隔，只有loss小于之前的最小值时才会保存

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_batch, train_label_batch=data_preprocess.get_batch(train,
                                train_label,
                                IMG_W,
                                IMG_H,
                                BATCH_SIZE,
                                CAPACITY,
                                is_training=True,
                                mode=FLAGS.data_preprocess)
# 将输入命名以便在Android app中使用
train_batch = tf.add(train_batch, tf.zeros([IMG_W, IMG_H, 3]), name="input")
one_hot_labels = tf.one_hot(indices=tf.cast(train_label_batch, tf.int32), depth=2)

# 获取输出
learning_rate = tf.placeholder(tf.float32)
train_op, train_loss, train_logits = resnet.inference(train_batch, one_hot_labels, DATA_SIZE, \
                                                                 deep=N_CLASSES, is_training=True, batch_size=BATCH_SIZE, lr=learning_rate, \
                                                                 net_mode=FLAGS.net_mode, opt_mode=FLAGS.opt_mode, lr_mode=FLAGS.lr_mode)
correct_prediction = tf.equal(tf.argmax(train_logits, 1), tf.argmax(one_hot_labels, 1))
train_acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
# 初始化
summary_op = tf.summary.merge_all() #这个是log汇总记录
sess = tf.Session()
train_writer = tf.summary.FileWriter(checkpoint_dir, sess.graph)
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator() #队列监控
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
total_begin_time = time.time()
time_list = []

# checkpoint读取
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
start_step = 0
if ckpt and ckpt.model_checkpoint_path:
    print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
    # Restores from checkpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
    
# 训练结果统计
tra_loss_list = [] #当前迭代的各step loss统计
tra_acc_list = [] #当前迭代的各step准确度统计
min_loss = 10000 #历次迭代的最小loss，设一个足够大的初始值
last_save_loss = 10000 #最后一次checkpoint保存时的loss，设一个足够大的初始值
loss_queue = Queue.Queue(EARLY_STOP_NUM) #最近的EARLY_STOP_NUM个迭代的loss统计，用于计算早期停止

# 自动衰减学习速率的相关变量
AUTO_LR_LOOP = 5 #多少次迭代内loss未下降时进行衰减
auto_lr = 0.001 #当前学习速率
auto_lr_min_loss = 10000 #AUTO_LR_LOOP次迭代前的loss最小值
auto_lr_queue = Queue.Queue(AUTO_LR_LOOP) #最近的AUTO_LR_LOOP个迭代的loss统计
auto_lr_loop_wait = AUTO_LR_LOOP #衰减冷却计数，每次衰减后会有AUTO_LR_LOOP的衰减冷却时间，以迭代次数为单位

# 周期性学习速率的相关变量
period_lr_T = 30 #重启周期，以迭代次数为单位
period_lr_T_now = 0 #当前周期的迭代次数

# 进行训练
last_time = time.time()
for step in np.arange(start_step + 1, MAX_STEP):
    if coord.should_stop():
        break
    _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc], feed_dict={learning_rate:auto_lr})
    tra_loss_list.append(tra_loss)
    tra_acc_list.append(tra_acc)
        
    # 每隔SAVE_INTERVAL步，检查是否要保存模型
    step = int(step)
    if step % SAVE_INTERVAL == 0 or (step + 1) == MAX_STEP:
        tra_loss_mean = np.mean(tra_loss_list)
        if tra_loss_mean < last_save_loss: #只有当前loss小于此前的最小值才进行保存
            last_save_loss = tra_loss_mean
            checkpoint_path = os.path.join(checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
    
    # 完成了一次迭代
    if step % (DATA_SIZE // BATCH_SIZE) == 0:
        # 当前迭代的平均loss和准确度计算
        tra_loss_mean = np.mean(tra_loss_list)
        tra_acc_mean = np.mean(tra_acc_list)
        tra_loss_list = []
        tra_acc_list = []
        
        #打印当前迭代的平均loss以及acc，同时记录log，写入writer
        loop = step / (DATA_SIZE // BATCH_SIZE)
        log_info = 'loop %d, Step %d (%.2fs), train loss = %.4f, train accuracy = %.2f%%' \
            %(loop, step, time.time() - last_time, tra_loss_mean, tra_acc_mean*100.0)
        time_list.append(time.time() - last_time)
        last_time = time.time()
        print(log_info)
        summary_str = sess.run(summary_op, feed_dict={learning_rate: auto_lr})
        train_writer.add_summary(summary_str, step)
        
        #学习速率自动衰减
        if FLAGS.lr_mode == 4:
            if auto_lr_queue.qsize() == AUTO_LR_LOOP:
                auto_lr_min_loss = min(auto_lr_min_loss, auto_lr_queue.get())
            auto_lr_queue.put(tra_loss_mean)
            if tra_loss_mean >= auto_lr_min_loss:
                if auto_lr_loop_wait == 0: #衰减冷却时间为0
                    auto_lr /= 2
                    print("learning_rate = %.6f" %(auto_lr))
                    auto_lr_loop_wait = AUTO_LR_LOOP #衰减冷却时间重新计数
                else:
                    auto_lr_loop_wait -= 1
            else:
                auto_lr_loop_wait = AUTO_LR_LOOP   
        #学习速率周期性重启，余弦下降，在[0 - 0.001]间取值
        elif FLAGS.lr_mode == 5:
            period_lr_T_now += 1
            auto_lr = 0.0005 + 0.0005 * math.cos(math.pi * period_lr_T_now / period_lr_T) #更新学习速率
            if period_lr_T_now == period_lr_T: #热重启
                period_lr_T_now = 0
    
        print("learning_rate = %.6f" %(auto_lr))
        
        # 早期停止，比较EARLY_STOP_NUM次迭代内，loss是否比EARLY_STOP_NUM次之前更低
        if loss_queue.qsize() == EARLY_STOP_NUM:
            min_loss = min(min_loss, loss_queue.get()) #EARLY_STOP_NUM次迭代之前的最低loss
        loss_queue.put(tra_loss_mean)
        early_stop = True
        temp_loss_queue = Queue.Queue(EARLY_STOP_NUM)
        while not loss_queue.empty(): #遍历最近的EARLY_STOP_NUM次迭代
            num = loss_queue.get()
            temp_loss_queue.put(num)
            if num < min_loss:
                early_stop = False
        loss_queue = temp_loss_queue
        if early_stop:
            log_info = 'early_stop!'
            print(log_info)
            with open(checkpoint_dir + "log.txt", "w") as log_file:
                log_file.write(log_info+'\n')
            break
            
print('total time cost = %.2f, mean time cost = %.2f' %(time.time() - total_begin_time, np.mean(time_list)))
#except tf.errors.OutOfRangeError:
#    print('Done training -- epoch limit reached')
#finally:
coord.request_stop()
    
output = tf.add(train_logits, tf.zeros(N_CLASSES), name="output")
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"])
with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    f.write(constant_graph.SerializeToString())
    