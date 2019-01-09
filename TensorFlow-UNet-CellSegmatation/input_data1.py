# from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def get_files(image_path, mask_path, shuffle=False):
    """
    从图片文件夹中生成图片路径列表
    :param image_path: 存放image的文件夹路径
    :param mask_path: 存放mask的文件夹路径
    :param shuffle: 是否打乱，默认否
    :return: image_list, mask_list
    """

    image_list = []
    mask_list = []

    for file in os.listdir(image_path):
        image_list.append(os.path.join(image_path, file))
    for file in os.listdir(mask_path):
        mask_list.append(os.path.join(mask_path, file))
    if len(image_list) == len(mask_list):
        print('there are %d images and %d masks' % (len(image_list), len(mask_list)))
    else:
        print('image number is not equal to mask')
        return
    if shuffle:
        index = np.arange(len(image_list))
        np.random.shuffle(index)
        image_list = np.array(image_list)[index]
        mask_list = np.array(mask_list)[index]

    return image_list, mask_list


def get_batch(image_list, mask_list, batch_size=2):
    """
    从图片路径列表中获取批量图片
    :param image_list: image路径李彪
    :param mask_list: mask路径列表
    :param batch_size: 批大小，默认为2
    :return:
    """
    # 转换为tf格式
    image_list = tf.cast(image_list, tf.string)
    mask_list = tf.cast(mask_list, tf.string)
    input_queue = tf.train.slice_input_producer([image_list, mask_list])
    # 对image进行处理
    image_constent = tf.read_file(input_queue[0])
    image = tf.image.decode_bmp(image_constent, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, 512, 512)
    image = tf.cast(image, tf.float32)/255
    # 对mask进行处理
    mask_constent = tf.read_file(input_queue[1])
    mask = tf.image.decode_bmp(mask_constent, channels=3)
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.image.resize_image_with_crop_or_pad(mask, 512, 512)
    mask = tf.cast(mask, tf.float32)/255

    image_batch, mask_batch = tf.train.shuffle_batch([image, mask],
                                                     batch_size=batch_size,
                                                     capacity=200,
                                                     min_after_dequeue=100,
                                                     num_threads=4)
    return image_batch, mask_batch


if __name__ == '__main__':
    # 测试程序
    image_path = 'dataset/train/Image/'
    mask_path = 'dataset/train/Mask/'
    image_list, mask_list = get_files(image_path, mask_path, shuffle=True)
    # print(mask_list)
    image_batch, mask_batch = get_batch(image_list, mask_list)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for i in range(1):
                if coord.should_stop():
                    break
                train_image_batch, train_mask_batch = sess.run([image_batch, mask_batch])
                plt.subplot(221)
                plt.imshow(train_image_batch[0])
                plt.subplot(222)
                plt.imshow(train_image_batch[1])
                plt.subplot(223)
                plt.imshow(train_mask_batch[0][:, :, 0], cmap='gray')
                plt.subplot(224)
                plt.imshow(train_mask_batch[1][:, :, 0], cmap='gray')
                plt.show()

        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            coord.request_stop()
            coord.join(threads=threads)
