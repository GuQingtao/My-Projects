import os
import numpy as np
import tensorflow as tf
from input_data1 import *
from AlexNet import *


N_CLASSES = 1000
IMG_SIZE = 227
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 1000
LEARNING_RATE = 1e-4

def train_model():
    """
    模型训练过程
    """
    train_dir = 'dataset/train/'
    train_label_file = 'dataset/train.txt'
    train_logs_dir = 'logs/train/'

    val_dir = 'dataset/val/'
    val_label_file = 'dataset/val.txt'
    val_logs_dir = 'logs/val/'
    models_dir = 'models/'

    # 批量获取训练集图片和标签
    train_image_list, train_label_list = get_files(train_dir, train_label_file)
    train_image_batch, train_label_batch = get_batch([train_image_list, train_label_list])

    # 批量获取验证集图片和标签
    val_image_list, val_label_list = get_batch(val_dir, val_label_file)
    val_image_batch, val_label_batch = get_batch(val_image_list, val_label_list)

    # 定义前向传播过程
    logits = inference(train_image_batch, TRAIN_BATCH_SIZE, N_CLASSES)
    loss = losses(logits, train_label_batch)
    train_op = training_optimizer(loss, LEARNING_RATE)
    acc = evaluation(logits, TRAIN_BATCH_SIZE)

    # 定义占位符
    x_train = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y_train = tf.placeholder(tf.int16, [None])
    x_val = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y_val = tf.placeholder(tf.int16, [None])

    # 开启回话
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())                         # 运行初始化
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()                                      # 设置多线程协调器
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)      # 开始队列运行器

    summary_op = tf.summary.merge_all()                                 # 汇总操作
    train_writer = tf.summary.FileWriter(train_logs_dir, sess.graph)    # 将训练的汇总写入train_logs_dir
    val_writer = tf.summary.FileWriter(val_logs_dir, sess.graph)        # 将验证的汇总写入val_logs_dir

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            train_images, train_labels = sess.run([train_image_batch])
            _, train_loss, train_acc = sess.run([train_op, loss, acc],feed_dict={x_train: train_images,
                                                                                 y_train: train_labels})
            # 计算训练损失和正确率
            if step % 50 == 0:
                print('step %d, train loss = %.6f, train accuracy = %.2f%%' % (step, train_loss, train_acc*100.0))

                train_summary = sess.run(summary_op)
                train_writer.add_summary(train_writer, step)

            if (step % 200 == 0) or (step+1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, acc], feed_dict={x_val: val_images,
                                                                     y_val: val_labels})
                print('** step %d, val loss = %.2f, val accuracy = %.2f%% **' % (step, val_loss, val_acc*100.0))
                val_summary = sess.run(summary_op)
                val_writer.add_summary(val_summary, step)

            if (step % 2000 == 0) or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(models_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, step)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train_model()




