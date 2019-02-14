import tarfile
import os
import pickle
import numpy as np


def get_data_cifar_100(data_dir, f_name, mode="r:gz"):
    tar_file = tarfile.open(os.path.join(data_dir, f_name), mode=mode)

    print("Loading training data....")
    train_file = tar_file.extractfile(f_name[:-7] + "/train")
    train_data = pickle.load(train_file, encoding="latin1")
    x_train = train_data["data"]
    y_train = np.asarray(train_data["fine_labels"])
    train_file.close()
    # print(x_train.shape, y_train.shape)
    print("Training Data Loading Complete!\n")

    print("Loading testing data....")
    test_file = tar_file.extractfile(f_name[:-7] + "/test")
    test_data = pickle.load(test_file, encoding="latin1")
    x_test = test_data["data"]
    y_test = np.asarray(test_data["fine_labels"])
    test_file.close()
    # print(x_test.shape, y_test.shape)
    # print(y_test[0])
    print("Testing Data Loading Complete!\n")

    print("Loading labels names....")
    label_name_file = tar_file.extractfile(f_name[:-7] + "/meta")
    label_names = pickle.load(label_name_file)
    label_names = label_names["fine_label_names"]
    label_name_file.close()
    # print(label_names)
    print("Label names loading complete!\n")

    return x_train, y_train, x_test, y_test, label_names


if __name__ == "__main__":
    DATA_DIR = "../Data/"
    file_name = "cifar-100-python.tar.gz"
    x_train, y_train, x_test, y_test, label_names = get_data_cifar_100(DATA_DIR, file_name)
    print(x_train.shape)
    print(x_test.shape)
