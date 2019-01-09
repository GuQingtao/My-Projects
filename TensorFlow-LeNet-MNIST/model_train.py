import os
import time
import numpy as np
from model import *
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def train(mnist):

    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name='x-input')
    y = tf.placeholder(tf.float32, [BATCH_SIZE, OUTPUT_NODE], name='y-input')
    # 前向传播结果
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    logits = inference(x, False, regularizer)
    # 计算损失
    loss = losses(logits, y, regularizer)
    # 计算准确率
    accuracy = evaluation(logits, y)

    # 设置指数衰减的学习率。
    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
    #                                            global_step,
    #                                            mnist.train.num_examples / BATCH_SIZE,
    #                                            LEARNING_RATE_DECAY)
    # 优化损失函数
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    # 初始化会话并开始训练过程。
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        merged = tf.summary.merge_all()
        train_summary_writer = tf.summary.FileWriter(logs_dir+'train/', sess.graph)
        validation_summary_writer = tf.summary.FileWriter(logs_dir+'/validation', sess.graph)

        max_train_step = train_set_num // BATCH_SIZE
        max_validation_step = validation_set_num // BATCH_SIZE

        validation_acc = []
        global_step = 0
        validation_step = 0
        for epoch in range(EPOCH):
            for train_step in range(max_train_step):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                xs_reshape = np.reshape(xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
                _, train_loss, train_acc = sess.run([train_op, loss, accuracy], feed_dict={x: xs_reshape, y: ys})
                if train_step % 50 == 0:
                    print("After %d training step(s), train loss = %.2f , train acc = %.2f%%"
                          % (global_step, train_loss, train_acc*100))
                    train_summary = sess.run(merged, feed_dict={x: xs_reshape, y: ys})
                    train_summary_writer.add_summary(train_summary, global_step=global_step)
                # 模型验证并保存
                if train_step == (max_train_step-1):
                    total_acc = 0
                    for i in range(max_validation_step):
                        validation_xs, validation_ys = mnist.validation.next_batch(BATCH_SIZE)
                        validation_xs_reshape = np.reshape(validation_xs, (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
                        validation_summary = sess.run(merged, feed_dict={x: validation_xs_reshape, y: validation_ys})
                        validation_summary_writer.add_summary(validation_summary, global_step=validation_step)
                        validation_step += 1

                        val_batch_acc = sess.run(accuracy, feed_dict={x: validation_xs_reshape, y: validation_ys})
                        total_acc += val_batch_acc
                    ave_acc = total_acc/max_validation_step
                    print('===== After %4d epoch, test acc = %.2f%%' % (epoch, ave_acc*100))
                    validation_acc.append(ave_acc)

                    saver.save(sess, os.path.join(models_dir, 'model.ckpt'), global_step=global_step)

                global_step += 1

        plt.figure()
        plt.plot(validation_acc)
        plt.title('test_acc = %.2f%%' % (validation_acc[-1] * 100))
        plt.savefig(fig_name)
        plt.show()

def eval(mnist):
    EVAL_INTERVAL_SECS = 300
    # 初始化图
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, 28, 28, NUM_CHANNELS], name='x-input')
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    validate_feed = {x: np.reshape(mnist.validation.images, [-1, 28, 28, 1]),
                     y: mnist.validation.labels}

    y_pred = inference(x, True, None)

    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(models_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
            print('After {} training steps,validation accuracy is {}'.format(global_step, accuracy_score))
        else:
            print('No checkpoint file found')
        time.sleep(EVAL_INTERVAL_SECS)


if __name__ == '__main__':

    INPUT_NODE = 784
    OUTPUT_NODE = 10

    IMAGE_SIZE = 28
    NUM_CHANNELS = 1

    BATCH_SIZE = 256
    TRAINING_STEPS = 1000

    LEARNING_RATE_BASE = 0.8
    LEARNING_RATE_DECAY = 0.99

    REGULARIZATION_RATE = 0.1
    MOVING_AVERAGE_DECAY = 0.99

    EPOCH = 1000
    models_dir = 'models/batchsize256_epoch'+str(EPOCH)+'/'
    logs_dir = 'logs/batchsize256_epoch'+str(EPOCH)+'/'
    fig_name = 'batchsize256_epoch'+str(EPOCH)+'.jpg'
    learning_rate = 1e-4

    mnist = input_data.read_data_sets(r'E:\My_DL_Project\dataSet\MNIST', one_hot=True)
    train_set_num = len(mnist.train.images)
    validation_set_num = len(mnist.validation.images)
    test_set_num = len(mnist.test.images)
    print('train set num: %5d, validation set num: %5d, test set num: %5d'
          % (train_set_num, validation_set_num, test_set_num))
    # print(mnist.validation.labels)
    train(mnist)
    # eval(mnist)

