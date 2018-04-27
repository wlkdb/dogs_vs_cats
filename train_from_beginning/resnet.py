# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:27:34 2017

@author: liguo

从头训练使用的Resnet_v2网络，包含了loss的计算
"""

import collections
import tensorflow as tf

slim = tf.contrib.slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    'A named tuple describing a ResNet block.'

def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)
    
def conv2d_same(inputs, num_outputs, kernel_size, stride, scope=None):
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1,
                           padding='SAME', scope=scope)
    else:
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], 
                                 [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                           padding='VALID', scope=scope)
        
@slim.add_arg_scope
def stack_blocks_dense(net, blocks, outputs_collections=None):
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            for i, unit in enumerate(block.args):
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    unit_depth, unit_depth_bottleneck, unit_stride = unit
                    net = block.unit_fn(net, depth=unit_depth,
                                        depth_bottleneck=unit_depth_bottleneck,
                                        stride=unit_stride)
            net = slim.utils.collect_named_outputs(outputs_collections,
                                                       sc.name, net)
    return net

def resnet_arg_scope(is_training=True, weight_decay=0.0001, batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5, batch_norm_scale=True):
    batch_norm_params = {
            'is_training': is_training,
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            }
    with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),    #!
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
            
@slim.add_arg_scope
def buildneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None,
               scope=None):
    with tf.variable_scope(scope, 'buildneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None, 
                                   scope='shortcut')
        
        residual = conv2d_same(preact, depth_bottleneck, 3, stride,
                               scope='conv1')
        residual = slim.conv2d(residual, depth, 3, stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv2')
        
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name,
                                                output)
    
@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, outputs_collections=None,
               scope=None):
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None, 
                                   scope='shortcut')
        
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = conv2d_same(residual, depth_bottleneck, 3, stride,
                               scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')
        
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name,
                                                output)
        
def resnet_v2(inputs, blocks, num_classes=None, global_pool=True,
              include_root_block=True, reuse=None, scope=None):
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, stack_blocks_dense],
                            outputs_collections=end_points_collection):
            net = inputs
            if include_root_block:
                with slim.arg_scope([slim.conv2d], activation_fn=None,
                                    normalizer_fn=None):
                    net = conv2d_same(net, 64, 7, stride=2, scope='conv1')
                net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = stack_blocks_dense(net, blocks)
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
            if num_classes is not None:
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                  normalizer_fn=None, scope='logits')
            end_points= slim.utils.convert_collection_to_dict(end_points_collection)
            if num_classes is not None:
                end_points['predications'] = slim.softmax(net, scope='predictions')
            return net, end_points
    
def resnet_v2_18(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_18'):
    blocks = [
          Block('block1', buildneck, [(64, 64, 1)] + [(64, 64, 2)]),
          Block('block2', buildneck, [(128, 128, 1)] + [(128, 128, 2)]),
          Block('block3', buildneck, [(256, 256, 1)] + [(256, 256, 2)]),
          Block('block4', buildneck, [(512, 512, 1)] * 2)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)    
        
def resnet_v2_34(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_34'):
    blocks = [
          Block('block1', buildneck, [(64, 64, 1)] + [(64, 64, 2)]),
          Block('block2', buildneck, [(128, 128, 1)] * 3 + [(128, 128, 2)]),
          Block('block3', buildneck, [(256, 256, 1)] * 5 + [(256, 256, 2)]),
          Block('block4', buildneck, [(512, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)    
        

def resnet_v2_50(inputs,
                 num_classes=None,
                 global_pool=True,
                 reuse=None,
                 scope='resnet_v2_50'):
    blocks = [
          Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
          Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
          Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 2)]),
          Block('block4', bottleneck, [(2048, 512, 1)] * 3)]
    return resnet_v2(inputs, blocks, num_classes, global_pool,
                     include_root_block=True, reuse=reuse, scope=scope)    
        

_WEIGHT_DECAY = 2e-4
# _WEIGHT_DECAY = 2e-3 #其它可以考虑的l2 loss权值
_MOMENTUM = 0.9
_INITIAL_LEARNING_RATE = 0.001

def inference(features, one_hot_labels, data_size, deep = 2, is_training = True, batch_size = 16, lr = _INITIAL_LEARNING_RATE, net_mode = 0, opt_mode = 1, lr_mode = 1):
    
    with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        if net_mode == 0:
            net, end_points = resnet_v2_18(features, deep)
        elif net_mode == 1:
            net, end_points = resnet_v2_34(features, deep)
        elif net_mode == 2:
            net, end_points = resnet_v2_50(features, deep)
    
    net = tf.reshape(net, [-1, deep])
    
    cross_entropy = tf.losses.softmax_cross_entropy(logits=net, onehot_labels=one_hot_labels)
    
    # cross_entropy统计
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    # loss计算
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()]) # + \
    # 其他可以考虑的loss附加项
#            0.01 * tf.nn.l2_loss(tf.get_default_graph().get_tensor_by_name("logits/weights:0")) 
#            0.01 * tf.nn.l1_loss(tf.get_default_graph().get_tensor_by_name("logits/weights:0"))
    
    if is_training:
        global_step = tf.train.get_or_create_global_step()
            
        if lr_mode == 0:
            learning_rate = 0.01
        elif lr_mode == 1:
            learning_rate = 0.001
        elif lr_mode == 2:
            learning_rate = 0.0001
        # 带预热的固定比率衰减学习速率
        elif lr_mode == 3:
            batches_per_epoch = int(data_size / batch_size)
            boundaries = [
                batches_per_epoch * epoch for epoch in [10, 40, 80, 120]] 
            values = [
                _INITIAL_LEARNING_RATE * decay for decay in [0.1, 1, 0.1, 0.01, 1e-3]]
            learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
        # 是否使用传入的可变学习速率
        elif lr_mode > 3:
            learning_rate = lr 
        
        print("learning_rate = ", learning_rate)
        tf.summary.scalar('learning_rate', learning_rate)
        
        # 优化器的选择
        if opt_mode == 0:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif opt_mode == 1:
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)
        elif opt_mode == 2:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
          
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None  
    return train_op, cross_entropy, net   
    