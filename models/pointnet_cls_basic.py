# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size):
    # 一个batch是一组不同对象的点云集合
    # pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, None, 4))
    # 每个点云对应一个标签
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

# def hand_feature(point_cloud):
#     l = tf.reduce_max(point_cloud[:,:,0]) - tf.reduce_min(point_cloud[:,:,0])
#     l = tf.fill([1, 1, 1], l)
#     w = tf.reduce_max(point_cloud[:,:,1]) - tf.reduce_min(point_cloud[:,:,1])
#     w = tf.fill([1, 1, 1], w)
#     h = tf.reduce_max(point_cloud[:,:,2]) - tf.reduce_min(point_cloud[:,:,2])
#     h = tf.fill([1, 1, 1], h)
#     imax = tf.reduce_max(point_cloud[:,:,3])
#     imax = tf.fill([1, 1, 1], imax)
#     imean = tf.reduce_mean(point_cloud[:,:,3])
#     imean = tf.fill([1, 1, 1], imean)
#     feat = tf.concat([l,w,h,imax,imean], axis=2)
#     return feat

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx4, output Bx5 """
    # batch_size = point_cloud.get_shape()[0].value
    # num_point = point_cloud.get_shape()[1].value
    batch_size = tf.shape(point_cloud)[0]
    num_point = tf.shape(point_cloud)[1]
    end_points = {}

    input_image = tf.expand_dims(point_cloud[:,:,:], -1)
    
    # Point functions (MLP implemented as conv2d)
    net = tf_util.conv2d(input_image, 64, [10,4],
                         padding='VALID', stride=[10,1],
                         bn=False, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.conv2d(net, 1024, [1,1],
                        padding='VALID', stride=[1,1],
                        bn=False, is_training=is_training,
                        scope='conv5', bn_decay=bn_decay, use_xavier=True)

    # Symmetric function: max pooling
    # net = tf_util.max_pool2d(net, [num_point,1],
    #                          padding='VALID', scope='maxpool')
    net = tf.reduce_max(net, axis=1, keepdims=False)

    # MLP on global point cloud vector
    # net = tf.reshape(net, [batch_size, -1])
    net = tf.reshape(net, [batch_size, 1024])

    net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.fully_connected(net, 128, bn=False, is_training=is_training,
                                  scope='fc3', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.fully_connected(net, 64, bn=False, is_training=is_training,
                                  scope='fc4', bn_decay=bn_decay, use_xavier=True)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 5, activation_fn=None, scope='fc5', use_xavier=True)
    # print('net.shape', net.shape) # debug
    return net, end_points


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


# if __name__=='__main__':
#     with tf.Graph().as_default():
#         inputs = tf.zeros((32,1024,3))
#         outputs = get_model(inputs, tf.constant(True))
#         print(outputs)
