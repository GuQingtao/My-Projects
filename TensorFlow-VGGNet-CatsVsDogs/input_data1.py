from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import cv2
import os


def get_all_files(file_path):
    """
    获取图片路径及其标签
    :param file_path: a sting, 图片所在目录
    :return:
    """
    image_list = []
    label_list = []

    cat_count = 0
    dog_count = 0
    for item in os.listdir(file_path):              # 文件路径：dataset\\train
        item_path = file_path + '\\' + item
        item_label = item.split('.')[0]             # 文件名形如  cat.0.jpg,只需要取第一个

        if os.path.isfile(item_path):
            image_list.append(item_path)
        else:
            raise ValueError('文件夹中有非文件项.')

        if item_label == 'cat':                     # 猫标记为'0'
            label_list.append(0)
            cat_count += 1
        else:                                       # 狗标记为'1'
            label_list.append(1)
            dog_count += 1
    print('数据集中有%d只猫,%d只狗.' % (cat_count, dog_count))
    # 打乱
    index = np.arange(len(image_list))
    np.random.shuffle(index)
    image_list = np.array(image_list)[index]
    label_list = np.array(label_list)[index]

    return image_list, label_list


def get_batch(data_list, image_size, batch_size):
    """
    获取训练批次
    :param train_list: 2-D list, [image_list, label_list]
    :param image_size: a int, 训练图像大小
    :param batch_size: a int, 每个批次包含的样本数量
    :return:
    """

    input_queue = tf.train.slice_input_producer(data_list, shuffle=False)

    # 从路径中读取图片
    image = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image, channels=3)                     # 这里是jpg格式
    image = tf.image.resize_images(image, [image_size, image_size])     # 重塑图片大小
    image = tf.cast(image, tf.float32) / 255.                           # 转换数据类型并归一化

    # 图片标签
    label = input_queue[1]
    label = tf.one_hot(label, 2)

    # 获取批次
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      capacity=200,
                                                      min_after_dequeue=100,
                                                      num_threads=4)

    return image_batch, label_batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 图片读取
    image_dir = 'E:\\My_DL_Project\\dataSet\\Cat VS Dogs\\train'
    image_list, label_list = get_all_files(image_dir)

    # 训练集图片读展示
    train_list = [image_list, label_list]
    image_train_batch, label_train_batch = get_batch(train_list, 256, 1)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in range(10):
            if coord.should_stop():
                break
            image_batch, label_batch = sess.run([image_train_batch, label_train_batch])
            if label_batch[0, 0] == 1:
                label = 'Cat'
            else:
                label = 'Dog'
            plt.imshow(image_batch[0]), plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()
        coord.join(threads=threads)


