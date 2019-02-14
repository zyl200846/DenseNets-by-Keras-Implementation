"""
This Python script has following three functions:
    1. Randomly visualize 16 images from Cifar-100
    2. Visualizing one image from Cifar-100
    3. Generating the convolution results from a two-step
       convolution operations and one max-pooling results
       to see how convolution and max-pooling work
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Preprocess.normalize import normalize
from Preprocess.load_data import get_data_cifar_100
from Preprocess.utils import random_visualized_imgs, show_weights, show_conv_results

DATA_DIR = "./Data/"
file_name = "cifar-100-python.tar.gz"

# Get data and labels
x_train, y_train, x_test, y_test, label_names = get_data_cifar_100(DATA_DIR, file_name)
x_train = normalize(x_train)

# Randomly visualize the images from Cifar-100
random_visualized_imgs(x_train, y_train, label_names)

raw_data = x_train[0, :]
raw_img = np.reshape(raw_data, (24, 24))
plt.figure()
plt.imshow(raw_img, cmap='Greys_r')
plt.savefig('./Pictures/example_image.png')

k = 2
x = tf.reshape(raw_data, shape=[-1, 24, 24, 1])
x = tf.cast(x, dtype='float32')
b = tf.Variable(tf.random_normal([32]))
W = tf.Variable(tf.random_normal([5, 5, 1, 32]))

conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
conv_with_b = tf.nn.bias_add(conv, b)
conv_out = tf.nn.relu(conv_with_b)
max_pooling = tf.nn.max_pool(conv_out, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    W_val = sess.run(W)
    show_weights(W_val, './Pictures/step0_weights.png')

    conv_val = sess.run(conv)
    show_conv_results(conv_val, './Pictures/step1_conv.png')
    print(np.shape(conv_val))

    conv_out_val = sess.run(conv_out)
    show_conv_results(conv_out_val, './Pictures/step2_conv_outs.png')
    print(np.shape(conv_out_val))

    maxpool_val = sess.run(max_pooling)
    show_conv_results(maxpool_val, './Pictures/step3_maxpool.png')
    print(np.shape(maxpool_val))
