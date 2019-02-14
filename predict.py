import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import cifar100
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from Preprocess.load_data import get_data_cifar_100
from Preprocess.utils import normalize


json_file = open("./model_aug.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("./model_aug_weights.h5")

loaded_model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
DATA_DIR = "./Data/"
file_name = "cifar-100-python.tar.gz"
train_X, y_train_true, test_X, y_test_true, label_names = get_data_cifar_100(DATA_DIR, file_name)
# test_truth = y_test
# x_train = normalize(x_train)
# x_test = normalize(x_test)
# x_train = np.reshape(x_train, (-1, 24, 24, 1))
# x_test = np.reshape(x_test, (-1, 24, 24, 1))
print(y_train_true.shape)
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Keep ground truth for comparison between predictions and ground truth
train_truth = y_train.reshape((y_train.shape[0],))
test_truth = y_test.reshape((y_test.shape[0],))

print(train_truth)
print(test_truth.shape)
# Normalize data by subtract data via mean and multiply by scale factor
x_train[:, :, :, 0] = x_train[:, :, :, 0] - 123.68
x_train[:, :, :, 1] = x_train[:, :, :, 1] - 116.78
x_train[:, :, :, 2] = x_train[:, :, :, 2] - 103.94
x_train = x_train * 0.017

x_test[:, :, :, 0] = x_test[:, :, :, 0] - 123.68
x_test[:, :, :, 1] = x_test[:, :, :, 1] - 116.78
x_test[:, :, :, 2] = x_test[:, :, :, 2] - 103.94
x_test = x_test * 0.017

y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)
score = loaded_model.evaluate(x_test, y_test)
y_pred = np.argmax(loaded_model.predict(x_test), axis=1)
precision = precision_score(test_truth, y_pred, average="macro")
recall = recall_score(test_truth, y_pred, average="macro")
gt = list()
preds = list()
for i in range(10000):
    gt.append(label_names[test_truth[i]])
    preds.append(label_names[y_pred[i]])
# print(confusion_matrix(y_true=gt, y_pred=preds, labels=label_names))
df_cm = pd.DataFrame(confusion_matrix(y_true=gt, y_pred=preds, labels=label_names), index=label_names, columns=label_names)
df_cm.to_csv("confusion_matrix_aug.csv")
print(df_cm)
print(y_pred)
print(test_truth)
print(score)
print(precision, recall)
print(label_names)
