# -*- coding: utf-8 -*-
"""
Created on Mon May    7 18:14:37 2018

@author: liguo
迁移学习的训练
"""

import tensorflow as tf
import time
import os
from datasets import dataset_utils
from deployment import model_deploy
from preprocessing import vgg_preprocessing
from nets import nets_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_dir', '/tmp/tfmodel/',
                           'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1, 'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer('num_ps_tasks', 0,
                            'The number of parameter servers. If the value is 0, then the parameters '
                            'are handled locally by the worker.')

tf.app.flags.DEFINE_integer('num_readers', 4,
                            'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4,
                            'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer('log_every_n_steps', 10,
                            'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer('save_interval_secs', 600,
                            'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string('optimizer', 'rmsprop', '"adam" or "rmsprop".')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_string('dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('dataset_size', 1500, 'The size of dataset.')

tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left '
                           'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None, 'The maximum number of training steps.')


tf.app.flags.DEFINE_string('checkpoint_path', None, 'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,
                           'Comma-separated list of scopes of variables to exclude when restoring '
                           'from a checkpoint.')

tf.app.flags.DEFINE_string('trainable_scopes', None,
                           'Comma-separated list of scopes to filter the set of variables to train.'
                           'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean('ignore_missing_vars', False,
                            'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

NUM_CLASS = 2

# 获取训练数据集
def get_dataset(dataset_dir):
    file_pattern = 'train_*.tfrecord'
    file_pattern = os.path.join(dataset_dir, file_pattern)
    reader = tf.TFRecordReader
    keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                    [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)
    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'input data',
        'label': 'cat or dog',
    }
    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=reader,
            decoder=decoder,
            num_samples=FLAGS.dataset_size,
            items_to_descriptions=_ITEMS_TO_DESCRIPTIONS, 
            num_classes=NUM_CLASS,
            labels_to_names=labels_to_names)

# 选择 adam 或 rmsprop 优化器
def get_optimizer(learning_rate):
    if FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1)
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=0.9, epsilon=1)
    else:
        raise ValueError('优化器 %s 无法识别', FLAGS.optimizer)
    return optimizer

# 决定网络中的哪些参数需要进行训练
def get_variables_to_train():
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train

# 初始化网络
def get_init_fn():
    if FLAGS.checkpoint_path is None:
        return None

    # 根据train_dir中是否存在checkpoint文件来决定初始化方式
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info('checkpoint 已存在于 %s，忽略 --checkpoint_path 参数 '
                        % FLAGS.train_dir)
        return None

    # 根据 checkpoint_exclude_scopes参数 决定网络中哪些变量无需恢复
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    tf.logging.info('Fine-tuning from %s' % checkpoint_path)
    return slim.assign_from_checkpoint_fn(
            checkpoint_path,
            variables_to_restore,
            ignore_missing_vars=FLAGS.ignore_missing_vars)

def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('必须设置 dataset_dir 参数')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig(
                num_clones=FLAGS.num_clones,
                clone_on_cpu=FLAGS.clone_on_cpu,
                num_replicas=FLAGS.worker_replicas,
                num_ps_tasks=FLAGS.num_ps_tasks)

        with tf.device(deploy_config.variables_device()):
            global_step = slim.create_global_step()

        # 加载数据集，获取预处理和网络模型
        dataset = get_dataset(FLAGS.dataset_dir)
        network_fn = nets_factory.get_network_fn(
                FLAGS.model_name, num_classes=NUM_CLASS, weight_decay=FLAGS.weight_decay, is_training=True)

        # 根据 dataset 创建相应的 dataset provider
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size)
            [image, label] = provider.get(['image', 'label'])
            train_image_size = FLAGS.train_image_size or network_fn.default_image_size
            image = vgg_preprocessing.preprocess_image(
                    image, train_image_size, train_image_size, is_training=True)
            images, labels = tf.train.batch(
                    [image, label],
                    batch_size=FLAGS.batch_size,
                    num_threads=FLAGS.num_preprocessing_threads,
                    capacity=5 * FLAGS.batch_size)
            labels = slim.one_hot_encoding(labels, NUM_CLASS)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                    [images, labels], capacity=2 * deploy_config.num_clones)

        # 定义模型
        def clone_fn(batch_queue):
            images, labels = batch_queue.dequeue()
            logits, end_points = network_fn(images)
            if 'AuxLogits' in end_points:
                slim.losses.softmax_cross_entropy(
                        end_points['AuxLogits'], labels, weights=0.4,
                        scope='aux_loss')
            slim.losses.softmax_cross_entropy(logits, labels, weights=1.0)
            return end_points
        
        clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
        first_clone_scope = deploy_config.clone_scope(0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
        with tf.device(deploy_config.optimizer_device()):
            learning_rate = tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
            optimizer = get_optimizer(learning_rate)
            
        # 定义训练的tensor
        variables_to_train = get_variables_to_train()
        total_loss, clones_gradients = model_deploy.optimize_clones(
                clones, optimizer, var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(clones_gradients, global_step=global_step)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_tensor = tf.identity(total_loss, name='train_op')

       # 开始训练 
        session_config = tf.ConfigProto(allow_soft_placement=True) 
        start_time = time.time()
        slim.learning.train(
                train_tensor,
                logdir=FLAGS.train_dir,
                is_chief=True,
                init_fn=get_init_fn(),
                number_of_steps=FLAGS.max_number_of_steps,
                log_every_n_steps=FLAGS.log_every_n_steps,
                save_interval_secs=FLAGS.save_interval_secs,
                sync_optimizer=None,
                session_config=session_config) 
        print('total time cost = %.2f' %(time.time() - start_time))
        
if __name__ == '__main__':
    tf.app.run()
