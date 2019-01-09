import time
from load_data import *
from model import *
import matplotlib.pyplot as plt


# 训练模型
def training():
    N_CLASSES = 2
    IMG_SIZE = 208
    BATCH_SIZE = 64
    CAPACITY = 200
    MAX_EPOCH = 1
    # MAX_STEP = 10000
    LEARNING_RATE = 1e-4

    # 准备数据
    train_image_list, train_label_list, validation_image_list, validation_label_list \
        = get_all_files(image_dir, True)
    train_list = [train_image_list, train_label_list]
    validation_list = [validation_image_list, validation_label_list]
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    validation_image, validation_label = get_validation(validation_list, IMG_SIZE)

    # 前向传播
    # 占位符方法
    x = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
    y = tf.placeholder(tf.int32, [None, ])

    regularizer = tf.contrib.layers.l2_regularizer(LEARNING_RATE)
    train_logits = inference(x, regularizer, N_CLASSES)
    train_loss = losses(train_logits, y)
    train_acc = evaluation(train_logits, y)
    # 数据直接传入

    # train_logits = inference(image_train_batch, N_CLASSES)
    # 计算损失
    # train_loss = losses(train_logits, label_train_batch)
    # 计算准确率
    # train_acc = evaluation(train_logits, label_train_batch)
    # 优化
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    sess = tf.Session()
    var_list = tf.trainable_variables()
    paras_count = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_list])
    print('参数数目:%d' % sess.run(paras_count), end='\n\n')

    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    global_step = 0
    val_acc = []
    try:
        global_step = 0
        end_step = int(10000 / BATCH_SIZE)
        roclabel = []
        roclogits = []
        for epoch in range(MAX_EPOCH):
            for step in range(end_step):
                if coord.should_stop():
                    break

                # image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)

                xs, ys = sess.run([image_train_batch, label_train_batch])
                print(ys.shape)
                _, loss, acc = sess.run([train_op, train_loss, train_acc], feed_dict={x: xs, y: ys})
                # _, loss, acc = sess.run([train_op, train_loss, train_acc])
                # 每100步，实时记录训练过程并显示
                if global_step % 100 == 0:
                    runtime = time.time() - s_t
                    print('global step: %6d, loss: %.8f, accuracy: %.2f%%, time:%.2fs'
                          % (global_step, loss, acc * 100, runtime))
                    s_t = time.time()

                    train_summary = sess.run(merged, feed_dict={x: xs, y: ys})
                    # train_summary = sess.run(merged)
                    summary_writer.add_summary(train_summary, global_step)

                # 每个epoch结束后显示并保存的验证集准确率
                if (global_step+1) % 156 == 0:
                    total_acc = 0.0
                    for i in range(25):
                        val_xs, val_ys = sess.run([validation_image, validation_label])
                        total_acc += sess.run(train_acc, feed_dict={x: val_xs, y: val_ys})*100
                    val_acc.append(total_acc / 2500)
                    # 提前停止
                    val_acc_max = max(val_acc)
                    if (len(val_acc) > 10) and (val_acc[-10:] <= [val_acc_max]):
                        break

                global_step += 1

            # 保存每个epoch结束后的模型
            checkpoint_path = os.path.join(models_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()
    coord.join(threads=threads)
    sess.close()

    print('mean_val_acc_10:', tf.reduce_mean(val_acc[-10:]),
          'mean_val_acc_all:', tf.reduce_mean(val_acc[:]))
    plt.figure(1)
    plt.plot(val_acc)
    plt.title('验证集准确率')
    plt.show()


# 测试检查点
def model_test():
    N_CLASSES = 2
    IMG_SIZE = 208
    BATCH_SIZE = 1
    CAPACITY = 200
    MAX_STEP = 10

    test_dir = 'dataset\\test'
    models_dir = 'models'     # 检查点目录

    sess = tf.Session()

    train_list = get_all_files(test_dir, is_random=True)
    image_train_batch, label_train_batch = get_batch(train_list, IMG_SIZE, BATCH_SIZE, CAPACITY, True)
    train_logits = inference(image_train_batch, False, N_CLASSES)
    train_logits = tf.nn.softmax(train_logits)  # 用softmax转化为百分比数值

    # 载入检查点
    saver = tf.train.Saver()
    print('\n载入检查点...')
    ckpt = tf.train.get_checkpoint_state(models_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功，global_step = %s\n' % global_step)
    else:
        print('没有找到检查点')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image, prediction = sess.run([image_train_batch, train_logits])
            max_index = np.argmax(prediction)
            if max_index == 0:
                label = '%.2f%% is a cat.' % (prediction[0][0] * 100)
            else:
                label = '%.2f%% is a dog.' % (prediction[0][1] * 100)

            plt.imshow(image[0])
            plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    # 测试图片读取
    image_dir = 'dataset\\train'
    models_dir = 'models\\model1\\'
    logs_dir = 'logs\\log1\\'  # 检查点保存路径
    training()
    # model_test()
