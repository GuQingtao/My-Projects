# coding: utf-8
import tensorflow as tf

NUM_LABELS = 10
FC1_SIZE = 512


def inference(x, train, regulartizer):
    """LeNet5 with l2_regularizer"""
    with tf.variable_scope('layer1_conv1_pool1'):
        conv1_weight = tf.get_variable('weight',
                                       [5, 5, 1, 32],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_bias = tf.get_variable('bias',
                                     [32],
                                     initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer2_conv2_pool2'):
        conv2_weight = tf.get_variable('weight',
                                       [5, 5, 32, 64],
                                       initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_bias = tf.get_variable('bias',
                                     [64],
                                     initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    pool2_shape = pool2.get_shape().as_list()
    node = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    # print(pool2_shape)
    pool2_reshaped = tf.reshape(pool2, [-1, node])

    with tf.variable_scope('layer3_fc1'):
        fc1_weight = tf.get_variable('weight',
                                     [node, FC1_SIZE],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_bias = tf.get_variable('bias',
                                   [FC1_SIZE],
                                   initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool2_reshaped, fc1_weight) + fc1_bias)

        if regulartizer:
            tf.add_to_collection('losses', regulartizer(fc1_weight))
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer4_fc1'):
        fc2_weight = tf.get_variable('weight',
                                     [FC1_SIZE, NUM_LABELS],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_bias = tf.get_variable('bias',
                                   [NUM_LABELS],
                                   initializer=tf.constant_initializer(0.1))
        logits = tf.matmul(fc1, fc2_weight) + fc2_bias

    return logits


def losses(logits, labels, regularizer):
    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if regularizer:
            loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        else:
            loss = cross_entropy_mean
        tf.summary.scalar('loss', loss)
    return loss


def evaluation(logits, labels):
    with tf.variable_scope('accuracy'):
        correct = tf.cast(tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar('accuracy', accuracy)
    return accuracy


