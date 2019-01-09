from layer import *


def inference(x, img_channel, phase, drop_prob, n_class=1):
    """前向传播推导"""
    with tf.variable_scope('Conv_unit1'):
        with tf.variable_scope('conv1'):
            weights1_1 = weights_init([3, 3, img_channel, 32])
            biases1_1 = biases_init([32])
            conv1_1 = conv2d(x, weights1_1) + biases1_1
            bn1_1 = tf.contrib.layers.batch_norm(conv1_1, center=True, scale=True, is_training=phase, scope='bn1')
            relu1_1 = tf.nn.relu(bn1_1, name='relu')
            dropout1_1 = tf.nn.dropout(relu1_1, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights1_2 = weights_init([3, 3, 32, 32])
            biases1_2 = biases_init([32])
            conv1_2 = conv2d(dropout1_1, weights1_2) + biases1_2
            bn1_2 = tf.contrib.layers.batch_norm(conv1_2, center=True, scale=True, is_training=phase, scope='bn2')
            relu1_2 = tf.nn.relu(bn1_2, name='relu')
            dropout1_2 = tf.nn.dropout(relu1_2, drop_prob, name='dropout')

        with tf.variable_scope('pool'):
            pool1 = max_pool(dropout1_2)

    with tf.variable_scope('Conv_unit2'):
        with tf.variable_scope('conv1'):
            weights2_1 = weights_init([3, 3, 32, 64])
            biases2_1 = biases_init([64])
            conv2_1 = conv2d(pool1, weights2_1) + biases2_1
            bn2_1 = tf.contrib.layers.batch_norm(conv2_1, center=True, scale=True, is_training=phase, scope='bn1')
            relu2_1 = tf.nn.relu(bn2_1, name='relu')
            dropout2_1 = tf.nn.dropout(relu2_1, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights2_2 = weights_init([3, 3, 64, 64])
            biases2_2 = biases_init([64])
            conv2_2 = conv2d(dropout2_1, weights2_2) + biases2_2
            bn2_2 = tf.contrib.layers.batch_norm(conv2_2, center=True, scale=True, is_training=phase, scope='bn2')
            relu2_2 = tf.nn.relu(bn2_2, name='relu')
            dropout2_2 = tf.nn.dropout(relu2_2, drop_prob, name='dropout')

        with tf.variable_scope('pool'):
            pool2 = max_pool(dropout2_2)

    with tf.variable_scope('Conv_unit3'):
        with tf.variable_scope('conv1'):
            weights3_1 = weights_init([3, 3, 64, 128])
            biases3_1 = biases_init([128])
            conv3_1 = conv2d(pool2, weights3_1) + biases3_1
            bn3_1 = tf.contrib.layers.batch_norm(conv3_1, center=True, scale=True, is_training=phase, scope='bn1')
            relu3_1 = tf.nn.relu(bn3_1, name='relu')
            dropout3_1 = tf.nn.dropout(relu3_1, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights3_2 = weights_init([3, 3, 128, 128])
            biases3_2 = biases_init([128])
            conv3_2 = conv2d(dropout3_1, weights3_2) + biases3_2
            bn3_2 = tf.contrib.layers.batch_norm(conv3_2, center=True, scale=True, is_training=phase, scope='bn2')
            relu3_2 = tf.nn.relu(bn3_2, name='relu')
            dropout3_2 = tf.nn.dropout(relu3_2, drop_prob, name='dropout')

        with tf.variable_scope('pool'):
            pool3 = max_pool(dropout3_2)

    with tf.variable_scope('Conv_unit4'):
        with tf.variable_scope('conv1'):
            weights4_1 = weights_init([3, 3, 128, 256])
            biases4_1 = biases_init([256])
            conv4_1 = conv2d(pool3, weights4_1) + biases4_1
            bn4_1 = tf.contrib.layers.batch_norm(conv4_1, center=True, scale=True, is_training=phase, scope='bn1')
            relu4_1 = tf.nn.relu(bn4_1, name='relu')
            dropout4_1 = tf.nn.dropout(relu4_1, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights4_2 = weights_init([3, 3, 256, 256])
            biases4_2 = biases_init([256])
            conv4_2 = conv2d(dropout4_1, weights4_2) + biases4_2
            bn4_2 = tf.contrib.layers.batch_norm(conv4_2, center=True, scale=True, is_training=phase, scope='bn2')
            relu4_2 = tf.nn.relu(bn4_2, name='relu')
            dropout4_2 = tf.nn.dropout(relu4_2, drop_prob, name='dropout')

        with tf.variable_scope('pool'):
            pool4 = max_pool(dropout4_2)

    with tf.variable_scope('Conv_unit5'):
        with tf.variable_scope('conv1'):
            weights5_1 = weights_init([3, 3, 256, 512])
            biases5_1 = biases_init([512])
            conv5_1 = conv2d(pool4, weights5_1) + biases5_1
            bn5_1 = tf.contrib.layers.batch_norm(conv5_1, center=True, scale=True, is_training=phase, scope='bn1')
            relu5_1 = tf.nn.relu(bn5_1, name='relu')
            dropout5_1 = tf.nn.dropout(relu5_1, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights5_2 = weights_init([3, 3, 512, 512])
            biases5_2 = biases_init([512])
            conv5_2 = conv2d(dropout5_1, weights5_2) + biases5_2
            bn5_2 = tf.contrib.layers.batch_norm(conv5_2, center=True, scale=True, is_training=phase, scope='bn2')
            relu5_2 = tf.nn.relu(bn5_2, name='relu')
            dropout5_2 = tf.nn.dropout(relu5_2, drop_prob, name='dropout')

        with tf.variable_scope('deconv1'):
            weights5_3 = weights_init([3, 3, 256, 512])
            biases5_3 = biases_init([256])
            dconv1 = deconv2d(dropout5_2, weights5_3) + biases5_3

    with tf.variable_scope('Conv_unit6'):
        dconv_concat1 = crop_and_concat(dropout4_2, dconv1)
        with tf.variable_scope('conv1'):
            weights = weights_init([3, 3, 512, 256])
            biases = biases_init([256])
            conv = conv2d(dconv_concat1, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn1')
            relu = tf.nn.relu(bn, name='relu')
            dropout6_1 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights = weights_init([3, 3, 256, 256])
            biases = biases_init([256])
            conv = conv2d(dropout6_1, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn2')
            relu = tf.nn.relu(bn, name='relu')
            dropout5_2 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('deconv1'):
            weights = weights_init([3, 3, 128, 256])
            biases = biases_init([128])
            dconv1 = deconv2d(dropout5_2, weights) + biases

    with tf.variable_scope('Conv_unit7'):
        dconv_concat2 = crop_and_concat(dropout3_2, dconv1)
        with tf.variable_scope('conv1'):
            weights = weights_init([3, 3, 256, 128])
            biases = biases_init([128])
            conv = conv2d(dconv_concat2, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn1')
            relu = tf.nn.relu(bn, name='relu')
            dropout7_1 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights = weights_init([3, 3, 128, 128])
            biases = biases_init([128])
            conv = conv2d(dropout7_1, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn2')
            relu = tf.nn.relu(bn, name='relu')
            dropout7_2 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('deconv1'):
            weights = weights_init([3, 3, 64, 128])
            biases = biases_init([64])
            dconv1 = deconv2d(dropout7_2, weights) + biases

    with tf.variable_scope('Conv_unit8'):
        dconv_concat3 = crop_and_concat(dropout2_2, dconv1)
        with tf.variable_scope('conv1'):
            weights = weights_init([3, 3, 128, 64])
            biases = biases_init([64])
            conv = conv2d(dconv_concat3, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn1')
            relu = tf.nn.relu(bn, name='relu')
            dropout8_1 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights = weights_init([3, 3, 64, 64])
            biases = biases_init([64])
            conv = conv2d(dropout8_1, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn2')
            relu = tf.nn.relu(bn, name='relu')
            dropout8_2 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('deconv1'):
            weights = weights_init([3, 3, 32, 64])
            biases = biases_init([32])
            dconv1 = deconv2d(dropout8_2, weights) + biases

    with tf.variable_scope('Conv_unit9'):
        dconv_concat4 = crop_and_concat(dropout1_2, dconv1)
        with tf.variable_scope('conv1'):
            weights = weights_init([3, 3, 64, 32])
            biases = biases_init([32])
            conv = conv2d(dconv_concat4, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn1')
            relu = tf.nn.relu(bn, name='relu')
            dropout9_1 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('conv2'):
            weights = weights_init([3, 3, 32, 32])
            biases = biases_init([32])
            conv = conv2d(dropout9_1, weights) + biases
            bn = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn2')
            relu = tf.nn.relu(bn, name='relu')
            dropout9_2 = tf.nn.dropout(relu, drop_prob, name='dropout')

        with tf.variable_scope('output'):
            weights = weights_init([1, 1, 32, n_class])
            biases = biases_init([n_class])
            conv = conv2d(dropout9_2, weights) + biases
            output_map = tf.nn.sigmoid(conv, name='output')

    return output_map


def losses(logits, labels, loss_function='cross entropy'):
    """定义损失函数"""
    H, W, C = labels.get_shape().as_list()[1:]
    if loss_function == 'dice coefficient':
        # DICE相似系数
        smooth = 1e-5
        logits_reshape = tf.reshape(logits, [-1, H * W * C])
        labels_reshape = tf.reshape(labels, [-1, H * W * C])
        intersection = 2 * tf.reduce_sum(logits_reshape * labels_reshape, axis=1) + smooth
        denominator = tf.reduce_sum(logits_reshape, axis=1) + tf.reduce_sum(labels_reshape, axis=1) + smooth
        loss_dice = 1.0 - tf.reduce_mean(intersection / denominator)
        tf.summary.scalar('dice_loss', loss_dice)
        return loss_dice
    if loss_function == 'cross entropy':
        assert (C == 1)
        logits_reshape = tf.reshape(logits, [-1])
        labels_reshape = tf.reshape(labels, [-1])
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_reshape,
                                                                labels=labels_reshape)
        loss_cross_entopy = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('cross entropy loss', loss_cross_entopy)
        return loss_cross_entopy


def evaluation(logits, labels):
    """准确率"""
    H, W, C = labels.get_shape().as_list()[1:]
    smooth = 1e-5
    logits_reshape = tf.reshape(logits, [-1, H * W * C])
    labels_reshape = tf.reshape(labels, [-1, H * W * C])
    intersection = 2 * tf.reduce_sum(logits_reshape * labels_reshape, axis=1) + smooth
    denominator = tf.reduce_sum(logits_reshape, axis=1) + tf.reduce_sum(labels_reshape, axis=1) + smooth
    accuracy = tf.reduce_mean(intersection / denominator)

    tf.summary.scalar('accuracy', accuracy)
    return accuracy