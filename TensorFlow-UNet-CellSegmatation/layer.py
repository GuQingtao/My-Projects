import tensorflow as tf


def weights_init(shape):
    """权重初始化"""
    weights = tf.get_variable("weights", shape, tf.float32,
                              initializer=tf.truncated_normal_initializer(stddev=0.05))
    return weights


def biases_init(shape):
    """初始化偏置"""
    biases = tf.get_variable("biases", shape, tf.float32,
                             initializer=tf.constant_initializer(0.1))
    return biases


def conv2d(x, w, stride=1):
    """计算卷积"""
    conv = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME', name='conv')
    return conv


def max_pool(x):
    """计算池化"""
    pool = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',
                          name="max-pooling")
    return pool


def deconv2d(x, w, stride=2):
    """计算反卷积"""
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*stride, x_shape[2]*stride, x_shape[3]//stride])
    deconv = tf.nn.conv2d_transpose(x, w, output_shape,
                                    strides=[1, stride, stride, 1],
                                    padding='SAME',
                                    name="deconv")
    return deconv


def crop_and_concat(x1, x2):
    """x1正中心抽取一个与x2同样大小的张量，然后在第三个通道上组合"""
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)

    offset = [0, (x1_shape[1]-x2_shape[1])//2, (x1_shape[2]-x2_shape[2])//2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offset, size)
    return tf.concat([x1_crop, x2], 3)




