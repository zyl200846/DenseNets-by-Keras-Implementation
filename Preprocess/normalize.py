import numpy as np


def normalize(data):
    """
    This function is to gray_scale images and normalize images.
    :param data: the images data
    :return: normalized images
    """
    imgs = data.reshape(data.shape[0], 3, 32, 32)
    grayscale_imgs = imgs.mean(1)
    cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    img_data = cropped_imgs.reshape(data.shape[0], -1)
    img_size = np.shape(img_data)[1]
    means = np.mean(img_data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(img_data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1. / np.sqrt(img_size))
    normalized_img = (img_data - meansT) / adj_stds

    return normalized_img


def standard_normalize(data):
    """
    This function is to normalize images by mean and variance.
    :param data: the images data
    :return: normalized images
    """
    # imgs = data.reshape(data.shape[0], 3, 32, 32)
    # # grayscale_imgs = imgs.mean(1)
    # # cropped_imgs = grayscale_imgs[:, 4:28, 4:28]
    # # img_data = imgs.reshape(data.shape[0], -1)
    img_size = np.shape(data)[1]
    means = np.mean(data, axis=1)
    meansT = means.reshape(len(means), 1)
    stds = np.std(data, axis=1)
    stdsT = stds.reshape(len(stds), 1)
    adj_stds = np.maximum(stdsT, 1. / np.sqrt(img_size))
    normalized_img = (data - meansT) / adj_stds

    return normalized_img


def normalization(x_train, x_test):
    """
    Channel-wise normalization
    :param x_train: input training data
    :param x_test: test data
    :return: normalized results
    """

    X = np.vstack((x_train, x_test))
    n_channels = x_train.shape[3]
    for i in range(n_channels):
        mean = np.mean(X[:, :, :, i])
        std = np.std(X[:, :, :, i])
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean) / std
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean) / std

    return x_train, x_test
