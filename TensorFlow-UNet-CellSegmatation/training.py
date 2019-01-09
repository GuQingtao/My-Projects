from input_data1 import *
from model import *

IMG_WIDTH = 512
IMG_HEIGHT = 512
CHANNEL = 3
N_CLASS = 1

BATCH_SIZE = 4
TRAING_SET_NUM = 160
EPOCH = 1000

LEARNING_RATE = 1e-4
PHASE = 1
DROP_PROB = 0.5

def model_train():
    logs_dir = 'logs/log1/'
    models_dir = 'models/model1/'
    # 准备数据
    image_path = 'dataset/train/Image/'
    mask_path = 'dataset/train/Mask/'

    train_image_list, train_mask_list = get_files(image_path, mask_path)
    train_image_batch, train_mask_batch = get_batch(train_image_list, train_mask_list, BATCH_SIZE)
    # 前向传播
    x = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, CHANNEL])
    y = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, N_CLASS])
    logits = inference(x, CHANNEL, PHASE, DROP_PROB, N_CLASS)
    # 计算损失
    loss = losses(logits, y, loss_function='cross entropy')
    # 计算准确率
    accuracy = evaluation(logits, y)
    # 优化
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 保存日志
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    # 保存模型
    saver = tf.train.Saver()
    # 开启线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    global_step = 0
    max_step = TRAING_SET_NUM//BATCH_SIZE
    try:
        for epoch in range(EPOCH):
            for step in range(max_step):
                if coord.should_stop():
                    break
                xs, ys = sess.run([train_image_batch, train_mask_batch])
                _, train_loss, train_acc = sess.run([train_op, loss, accuracy],
                                                    feed_dict={x: xs, y: ys})
                # 显示训练过程
                if global_step % 100 == 0:
                    print('step: %6d, train_loss: %.4f, train_acc: %.2f%%'
                          % (global_step, train_loss, train_acc))
                    summary = sess.run(merged, feed_dict={x: xs, y: ys})
                    summary_writer.add_summary(summary, global_step)
                # 保存模型
                if (step+1) == max_step:
                    # print('=====epoch: %5d, train_loss: %.4f, train_acc: %.2f%%====='
                    #       % (epoch, train_loss, train_acc))
                    checkpoint_path = os.path.join(models_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step)
                global_step += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        coord.join(threads)


def model_test():
    tf.reset_default_graph()
    models_dir = 'models/model1/'
    # # 准备数据
    image_path = 'dataset/test/Image/'
    mask_path = 'dataset/test/Mask/'
    train_image_list, train_mask_list = get_files(image_path, mask_path)
    train_image_batch, train_mask_batch = get_batch(train_image_list, train_mask_list, batch_size=1)
    # 前向传播
    x = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, CHANNEL])
    y = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, N_CLASS])
    logits = inference(x, CHANNEL, PHASE, DROP_PROB, N_CLASS)
    accuracy = evaluation(logits, y)
    # 开启会话
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # 重新导入模型
    saver = tf.train.Saver()
    # saver.restore(sess, models_dir)
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(models_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')

    # 开启线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    try:
        for i in range(5):
            if coord.should_stop():
                break
            xs, ys = sess.run([train_image_batch, train_mask_batch])
            pred, test_acc = sess.run([logits, accuracy], feed_dict={x: xs, y: ys})
            print('test acc = %.2f%%' % test_acc)

            plt.figure(i)
            plt.subplot(131)
            plt.imshow(xs[0])
            plt.title('original')
            plt.subplot(132)
            plt.imshow(ys[0][:, :, 0], cmap='gray')
            plt.title('mask')
            plt.subplot(133)
            plt.imshow(pred[0][:, :, 0], cmap='gray')
            plt.title('pred')
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Error')
    finally:
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    # model_train()
    model_test()






