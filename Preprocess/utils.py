import numpy as np
import matplotlib.pyplot as plt
import random
from Preprocess.load_data import get_data_cifar_100
from Preprocess.normalize import normalize, normalization


def random_visualized_imgs(data, labels, label_names):
    """
    Randomly visualize the images from Cifar-100
    :param data: the images data
    :param labels: labels of corresponding images, type: int
    :param label_names: the names of the images, type: string
    :return: None
    """
    plt.figure()
    rows, cols = 4, 4
    random.seed(666)
    random_indices = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_indices[i]
        plt.title(label_names[labels[j]])
        img = np.reshape(data[j, :], (24, 24))
        plt.imshow(img, cmap='Greys_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('../Pictures/cifar-100-examples.png')


def visualize_color_imgs(data, labels, label_names):
    plt.figure()
    rows, cols = 4, 4
    random.seed(666)
    random_indices = random.sample(range(len(data)), rows * cols)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        j = random_indices[i]
        plt.title(label_names[labels[j]])
        im = data[j, :]
        im_r = im[0:1024].reshape(32, 32)
        im_g = im[1024:2048].reshape(32, 32)
        im_b = im[2048:].reshape(32, 32)
        img = np.dstack((im_r, im_g, im_b))
        plt.imshow(img)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('../Pictures/cifar-100-color-examples.png')


def show_weights(W, filename=None):
    """
    Visualize the weights that is used to conv the images to extract useful information
    from images such as detecting edges and shape of the object in the image
    :param W: the weights that are used to conv the image
    :param filename: the file name that is used to store the generated results
    :return: None
    """
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(W)[3]):
        img = W[:, :, 0, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()


def show_conv_results(data, filename=None):
    """
    Show the results of convolution operation
    :param data: the images data
    :param filename: the file name of stored picture
    :return: None
    """
    plt.figure()
    rows, cols = 4, 8
    for i in range(np.shape(data)[3]):
        img = data[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='Greys_r', interpolation='none')
        plt.axis('off')
        if filename:
            plt.savefig(filename)
        else:
            plt.show()


def reconstruct_images(data):
    x_train_new = list()
    for i in range(data.shape[0]):
        im = data[i, :]
        im_r = im[0:1024].reshape(32, 32)
        im_g = im[1024:2048].reshape(32, 32)
        im_b = im[2048:].reshape(32, 32)
        img = np.dstack((im_r, im_g, im_b))
        x_train_new.append(img)
    data = np.array(x_train_new)
    return data


if __name__ == '__main__':
    DATA_DIR = "../Data/"
    file_name = "cifar-100-python.tar.gz"

    # Get data and labels
    x_train, y_train, x_test, y_test, label_names = get_data_cifar_100(DATA_DIR, file_name)
    # x_tr_grey = normalize(x_train)
    # random_visualized_imgs(x_tr_grey, y_train, label_names)
    # print(x_train.shape)
    # visualize_color_imgs(x_train, y_train, label_names)
    x_train = reconstruct_images(x_train)
    x_test = reconstruct_images(x_test)
    print(x_train.shape)
    print(x_test.shape)
    x_train, x_test = normalization(x_train, x_test)
    print(x_train[0])
    # plt.imshow(x_train[2])
    # plt.show()
    # plt.axis("off")
    # print(label_names[y_train[2]])
