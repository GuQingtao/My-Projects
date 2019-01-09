import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers.python.layers import batch_norm
from  tensorflow.layers import dropout

def conv2d(name, input_data, out_channel):
    in_channel = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [3, 3, in_channel, out_channel], tf.float32,
                                  initializer=xavier_initializer_conv2d())
        biases = tf.get_variable('biases', [out_channel], tf.float32,
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_data, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        # 添加 BN 层
        bn = batch_norm(conv)
        act = tf.nn.relu(bn)
    return act


def max_pool(name, input_data):
    pool = tf.nn.max_pool(input_data, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    return pool


def fc(name, input_data, out_channel, activation=True, drop_prob=1, regularizer=False):
    input_data_shape = input_data.get_shape().as_list()
    if len(input_data_shape) == 4:
        size = input_data_shape[-1] * input_data_shape[-2] * input_data_shape[-3]
    else:
        size = input_data_shape[-1]
    reshaped_input_data = tf.reshape(input_data, [-1, size])
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', [size, out_channel], tf.float32,
                                  initializer=xavier_initializer_conv2d())
        # 加入正则化
        # tf.add_to_collection('losses', regularizer(weights))
        biases = tf.get_variable('biases', [out_channel], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc = tf.nn.bias_add(tf.matmul(reshaped_input_data, weights), biases)
        if activation:
            bn = batch_norm(fc)
            act = tf.nn.relu(bn)
            drop = dropout(act, drop_prob)
            return drop
        else:
            return fc

