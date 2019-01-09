import tensorflow as tf
import numpy as np
import os


def get_files(file_dir, label_file):
    image_list = []
    label_list = []

    len_file = len(os.listdir(file_dir))
    len_label = len(open(label_file).readlines())

    if len_file == len_label:
        print('num of image identify to num of labels')
        print('the len of file is %d' % len_file)

    txt_file = open(label_file, 'r')
    for file in os.listdir(file_dir):
        image_list.append(file_dir + file)
        one_content = txt_file.readline()
        name = one_content.split(' ')
        if name[0] == file:
            name1 = name[1].split('\n')
            label_list.append(int(name1[0])-1)
        else:
            print('File Name is different from label name!\n')
    txt_file.close()
    print('There are %d images\nThere are %d labels' % (len(image_list), len(label_list)))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = temp[:, 0]
    label_list = temp[:, 1]
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image_list, label_list, img_size, batch_size, capacity):
    image_cast = tf.cast(image_list, tf.string)
    label_cast = tf.cast(label_list, tf.int32)

    input_queue = tf.train.slice_input_producer([image_cast, label_cast])
    image_contents = tf.read_file(input_queue[0])
    images = tf.image.decode_jpeg(image_contents, channels=3)
    images = tf.image.resize_image_with_crop_or_pad(images, img_size, img_size)
    images = tf.image.per_image_standardization(images)

    labels = input_queue[1]

    image_batch, label_batch = tf.train.batch([images, labels],
                                              batch_size=batch_size,
                                              num_threads=64,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch

