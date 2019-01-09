from input_data1 import *
from VGG16_model import *
import matplotlib.pyplot as plt
# from tensorflow.contrib.layers import l2_regularizer

DATASET_NUM = 25000
TEST_SIZE = 0.2

IMAGE_CHANNEL = 3
N_CLASS = 2
IMAGE_SIZE = 256

REGULARIZER_RATE = 0.0001

def model_train():
    # 前向传播
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL], name='input')
    y = tf.placeholder(tf.int32, [None, N_CLASS], name='output')
    train_logits = inference(x, N_CLASS, train=True)
    # 计算损失
    # regularizer = l2_regularizer(REGULARIZER_RATE)
    # loss = losses(logits, y, regularizer)
    loss = losses(train_logits, y)
    accuracy = evaluation(train_logits, y)
    # 优化
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_summary_writer = tf.summary.FileWriter(test_log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    global_step = 0
    max_train_step = int(DATASET_NUM * (1 - TEST_SIZE) / BATCH_SIZE)
    max_test_step = int(DATASET_NUM * TEST_SIZE / BATCH_SIZE)
    test_acc = []

    try:
        for epoch in range(EPOCH):
            for step in range(max_train_step):
                if coord.should_stop():
                    break

                train_xs, train_ys = sess.run([train_image_batch, train_label_batch])
                _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                    feed_dict={x: train_xs, y: train_ys})

                if step % 100 == 0:
                    print('After %d steps, loss = %.4f, train_acc = %.2f%%' % (global_step, train_loss, train_acc*100))
                    train_summary = sess.run(merged, feed_dict={x: train_xs, y: train_ys})
                    train_summary_writer.add_summary(train_summary, global_step)

                if (step+1) == max_train_step:
                    total_acc = 0
                    for test_step in range(max_test_step):
                        test_xs, test_ys = sess.run([test_image_batch, test_label_batch])
                        test_batch_acc = sess.run(accuracy, feed_dict={x: test_xs, y: test_ys})

                        test_summary = sess.run(merged, feed_dict={x: test_xs, y: test_ys})
                        test_summary_writer.add_summary(test_summary, epoch*max_test_step+test_step)

                        total_acc += test_batch_acc
                    test_acc.append(total_acc/max_test_step)
                    print('======= %d epoch, test_acc = %.2f%%' % (epoch, test_acc[-1]*100))

                    saver.save(sess, os.path.join(model_dir, 'model.ckpt'), epoch)

                global_step += 1
    except tf.errors.OutOfRangeError:
        print('index out of range')
    finally:
        coord.request_stop()
        coord.join(threads)
    sess.close()

    plt.figure()
    plt.plot(test_acc)
    plt.savefig('epoch'+str(EPOCH)+'batch_size'+str(BATCH_SIZE) +'.jpg')
    plt.show()


def model_test_with_per_image():
    test_image, test_label = get_batch([test_image_list, test_label_list], IMAGE_SIZE, batch_size=1)
    test_logits = inference(test_image, N_CLASS, train=True)
    # 载入检查点
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        # plt.figure()
        for step in range(9):
            if coord.should_stop():
                break
            image, label, prediction = sess.run([test_image, test_label, tf.nn.softmax(test_logits)])
            print(label, prediction)
            max_index = np.argmax(prediction)

            if max_index == 0:
                label = '%.2f%% is a cat.' % (prediction[0][0]*100)
            else:
                label = '%.2f%% is a dog.' % (prediction[0][1]*100)
            plt.subplot(3, 3, step+1)
            plt.imshow(image[0])
            plt.title(label)
        plt.savefig('testfig_Epoch{}_BatchSize{}UsingOneSet.jpg'.format(EPOCH, BATCH_SIZE))
        plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()
        coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    BATCH_SIZE = 16
    EPOCH = 30
    LEARNING_RATE = 0.01

    image_dir = 'E:\\My_DL_Project\\dataSet\\Cat VS Dogs\\train'
    train_log_dir = 'logs/Epoch_{} BatchSize_{}_UsingOneSet/train/'.format(EPOCH, BATCH_SIZE)
    test_log_dir = 'logs/Epoch_{} BatchSize_{}_UsingOneSet/test/'.format(EPOCH, BATCH_SIZE)
    model_dir = 'models/Epoch_{} BatchSize_{}_UsingOneSet/'.format(EPOCH, BATCH_SIZE)

    # 准备数据
    np.random.seed(7)
    image_list, label_list = get_all_files(image_dir)
    train_image_list, test_image_list, train_label_list, test_label_list = \
        train_test_split(image_list, label_list, test_size=TEST_SIZE)
    train_image_batch, train_label_batch = get_batch([train_image_list, train_label_list], IMAGE_SIZE, BATCH_SIZE)
    test_image_batch, test_label_batch = get_batch([test_image_list, test_label_list], IMAGE_SIZE, BATCH_SIZE)
    # 训练
    # model_train()
    # 测试

    model_test_with_per_image()

