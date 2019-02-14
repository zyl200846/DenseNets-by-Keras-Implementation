import time
import os
import json
import numpy as np
from Preprocess.load_data import get_data_cifar_100
from Preprocess.normalize import normalization, normalize, standard_normalize
from Preprocess.utils import reconstruct_images
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from model import dense_net


# GET DATA
DATA_DIR = "./Data/"
file_name = "cifar-100-python.tar.gz"
x_train, y_train, x_test, y_test, label_names = get_data_cifar_100(DATA_DIR, file_name)
x_train = normalize(x_train)
x_test = normalize(x_test)
x_train = np.reshape(x_train, (-1, 24, 24, 1))
x_test = np.reshape(x_test, (-1, 24, 24, 1))
# x_train = reconstruct_images(x_train)
# x_test = reconstruct_images(x_test)
# x_train, x_test = normalization(x_train, x_test)
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# print(x_train[0].shape)
initial_learning_rate = 1e-3
batch_size = 64
model = dense_net(input_shape=(24, 24, 1), depth=40, num_dense_block=3, num_filters=16, growth_rate=12,
                  dropout_rate=0.2, concate_axis=-1, weight_decay=1e-4)

# plot_model(model, to_file="./Pictures/model_structure.png", show_shapes=True, show_layer_names=True)
# opt = SGD(lr=initial_learning_rate, momentum=0.85, nesterov=True)
model.compile(optimizer=Adam(lr=initial_learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
train_loss_list = list()
test_loss_list = list()
learning_rate_list = list()
print("Start Training...")
for i in range(300):
    if i == 150:
        K.set_value(model.optimizer.lr, np.float32(initial_learning_rate / 10.))
    elif i == 225:
        K.set_value(model.optimizer.lr, np.float32(initial_learning_rate / 100.))

    num_steps = x_train.shape[0] // batch_size
    split_indices = np.array_split(np.arange(x_train.shape[0]), num_steps)

    start_time = time.time()
    training_losses = list()
    training_acc = list()
    for batch_indices in split_indices:
        x_train_batch, y_train_batch = x_train[batch_indices], y_train[batch_indices]
        loss, acc = model.train_on_batch(x=x_train_batch, y=y_train_batch)
        training_losses.append(loss)
        training_acc.append(acc)

    test_loss, test_acc = model.evaluate(x=x_test, y=y_test, verbose=0, batch_size=64)
    train_loss_list.append(np.mean(np.array(training_losses), 0).tolist())
    test_loss_list.append([test_loss, test_acc])
    learning_rate_list.append(float(K.get_value(model.optimizer.lr)))
    print("epoch %s/%s, time: %s, train loss: %s, test loss: %s, "
          "train acc: %s, test acc: %s" % (i + 1, 300,
                                           time.time() - start_time,
                                           np.mean(np.array(training_losses), 0),
                                           test_loss,
                                           np.mean(np.array(training_acc)),
                                           test_acc))
    training_log = {}
    training_log["batch_size"] = batch_size
    training_log["epochs"] = 300
    training_log["optimizer"] = model.optimizer.get_config()
    training_log["train_loss"] = train_loss_list
    training_log["test_loss"] = test_loss_list
    training_log["lr"] = learning_rate_list
    LOG = './log'
    if not os.path.exists(LOG):
        os.makedirs(LOG)
    json_file_training = os.path.join(LOG, 'cifar-100.json')
    with open(json_file_training, "w") as fp:
        json.dump(training_log, fp, indent=4, sort_keys=True)

print("Saving model to JSON file and weights to HDF5...")
# Serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# Save weights to hdf5 file
model.save_weights("model_weights.h5")
