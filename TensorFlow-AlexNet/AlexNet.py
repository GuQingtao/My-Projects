import tensorflow as tf


def inference(images, batch_size, n_classes, train=True):
    """
    定义 AlexNet 网络结构
    包括：输入层 + 5个卷积层 + 3个全连接层
    """
    with tf.variable_scope('conv1+relu1+pool1+lrn1'):
        weights = tf.get_variable('weights', [11, 11, 3, 96], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [96, ], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv1 = tf.nn.conv2d(images, weights, strides=[1, 4, 4, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases))
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        lrn1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0, beta=0.75)

    with tf.variable_scope('conv2+relu2+pool2+lrn2'):
        weights = tf.get_variable('weights', [5, 5, 96, 256], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [256, ], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(lrn1, weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
        lrn2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    with tf.get_variable('con3+relu3'):
        weights = tf.get_variable('weights', [3, 3, 256, 384], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [384], tf.float32,
                                 initializer=tf.constant_initializer(0.01))
        conv3 = tf.nn.conv2d(lrn2, weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, biases))

    with tf.get_variable('con4+relu4'):
        weights = tf.get_variable('weights', [3, 3, 384, 384], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [384], tf.float32,
                                 initializer=tf.constant_initializer(0.01))
        conv4 = tf.nn.conv2d(relu3, weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, biases))

    with tf.variable_scope('conv5+relu5+pool5'):
        weights = tf.get_variable('weights', [3, 3, 384, 256], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable('biases', [256, ], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv5 = tf.nn.conv2d(lrn1, weights, strides=[1, 1, 1, 1], padding='SAME')
        relu5 = tf.nn.relu(tf.nn.bias_add(conv2, biases))
        pool5 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope('fc1+relu+dropout1'):
        pool5_reshaped = tf.reshape(pool5, [batch_size, -1])
        dim = pool5_reshaped.get_shape()[1].value
        weights = tf.get_variable('weights', [dim, 4096], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable('biases', [4096], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(pool5_reshaped, weights) + biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('fc2+relu+dropout2'):
        weights = tf.get_variable('weights', [4096, 4096], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable('biases', [4096], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('softmax_linear'):
        weights = tf.get_variable('weights', [4096, n_classes], tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005))
        biases = tf.get_variable('biases', [n_classes], tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.matmul(fc2, weights) + biases

    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss


def training_optimizer(loss, learning_rate):
    with tf.name_scope('optimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name+'/accuracy', accuracy)
    return accuracy
