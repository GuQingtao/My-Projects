import numpy as np
from layer import *


def inference(x, n_class, train=True):
    """ VGG16 结构 """
    with tf.variable_scope('CONV1'):
        conv1_1 = conv2d('conv1_1', x, 64)
        conv1_2 = conv2d('conv1_2', conv1_1, 64)
        pool1 = max_pool('pool1', conv1_2)

    with tf.variable_scope('CONV2'):
        conv2_1 = conv2d('conv2_1', pool1, 128)
        conv2_2 = conv2d('conv2_2', conv2_1, 128)
        pool2 = max_pool('pool2', conv2_2)

    with tf.variable_scope('CONV3'):
        conv3_1 = conv2d('conv3_1', pool2, 256)
        conv3_2 = conv2d('conv3_2', conv3_1, 256)
        conv3_3 = conv2d('conv3_3', conv3_2, 256)
        pool3 = max_pool('pool3', conv3_3)

    with tf.variable_scope('CONV4'):
        conv4_1 = conv2d('conv4_1', pool3, 512)
        conv4_2 = conv2d('conv4_2', conv4_1, 512)
        conv4_3 = conv2d('conv4_3', conv4_2, 512)
        pool4 = max_pool('pool4', conv4_3)

    with tf.variable_scope('CONV5'):
        conv5_1 = conv2d('conv5_1', pool4, 512)
        conv5_2 = conv2d('conv5_2', conv5_1, 512)
        conv5_3 = conv2d('conv5_3', conv5_2, 512)
        pool5 = max_pool('pool5', conv5_3)

    with tf.variable_scope('FC'):
        if train:
            fc1 = fc('fc1', pool5, 4096, activation=True, drop_prob=0.5)
            fc2 = fc('fc2', fc1, 4096, activation=True, drop_prob=0.5)
        else:
            fc1 = fc('fc1', pool5, 4096, activation=True)
            fc2 = fc('fc2', fc1, 4096, activation=True)

        fc3 = fc('fc3', fc2, n_class, activation=False)

    return fc3


def losses(logits, labels, regularizer=False):
    with tf.name_scope('losses'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(labels, 1))
        loss = tf.reduce_mean(cross_entropy)
        if regularizer:
            loss = loss + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('losses', loss)
    return loss


def evaluation(logits, labels):
    with tf.name_scope('accuracy'):
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.scalar('accuracy', accuracy)
    return accuracy

















