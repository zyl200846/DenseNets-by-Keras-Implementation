import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar100
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score
from model import dense_net
from Preprocess.load_data import get_data_cifar_100


# Hyper-parameters Settings
batch_size = 128
num_klasses = 100
input_shape = (32, 32, 3)
depth = 40
growth_rate = 12
num_filters = -1
dropout_rate = 0.0
n_dense_block = 3

# Create model
model = dense_net(input_shape=(32, 32, 3), depth=40, num_dense_block=3, num_filters=16, growth_rate=12,
                  dropout_rate=0.2, concate_axis=-1, weight_decay=1e-4)
print("Model created....")

model.compile(optimizer=Adam(lr=1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# Load data
DATA_DIR = "./Data/"
file_name = "cifar-100-python.tar.gz"
train_X, y_train_true, test_X, y_test_true, label_names = get_data_cifar_100(DATA_DIR, file_name)
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Keep ground truth for comparison between predictions and ground truth
train_truth = y_train
test_truth = y_test

# Normalize data by subtract data via mean and multiply by scale factor
x_train[:, :, :, 0] = x_train[:, :, :, 0] - 123.68
x_train[:, :, :, 1] = x_train[:, :, :, 1] - 116.78
x_train[:, :, :, 2] = x_train[:, :, :, 2] - 103.94
x_train = x_train * 0.017

x_test[:, :, :, 0] = x_test[:, :, :, 0] - 123.68
x_test[:, :, :, 1] = x_test[:, :, :, 1] - 116.78
x_test[:, :, :, 2] = x_test[:, :, :, 2] - 103.94
x_test = x_test * 0.017

print(x_train.shape)
print(x_test.shape)

# One hot encoding
y_train = to_categorical(y_train, num_classes=100)
y_test = to_categorical(y_test, num_classes=100)

# data augmentation
train_gen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)
train_gen.fit(x_train, seed=666)

# Save weights to file
weights_file = "./weights/DenseNet-CIFAR100.h5"
if not os.path.exists(os.path.dirname(weights_file)):
    os.makedirs(os.path.dirname(weights_file))

# Set call backs for learning rate and model check point
lr_reducer = ReduceLROnPlateau(monitor="val_acc", factor=np.sqrt(0.1), min_lr=1e-5, cooldown=0, patience=5)
model_checkpoint = ModelCheckpoint(weights_file, monitor="val_acc", save_best_only=True,
                                   save_weights_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_acc", patience=50)
callbacks = [lr_reducer, model_checkpoint, early_stop]

# Start Training model
model.fit_generator(train_gen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=300, callbacks=callbacks, validation_data=(x_test, y_test),
                    validation_steps=x_test.shape[0] // batch_size, verbose=1)

# Generate predictions and get accuracy, precision and recall
y_pred = np.argmax(model.predict(x_test), axis=1)
accuracy = accuracy_score(test_truth, y_pred)
precision = precision_score(test_truth, y_pred, average="macro")
recall = recall_score(test_truth, y_pred, average="macro")
print("Accuracy is: %s, Precision: %s, Recall: %s" % (accuracy, precision, recall))

print("Saving model to JSON file and weights to HDF5...")
# Serialize model to JSON
model_json = model.to_json()
with open("model_aug.json", "w") as json_file:
    json_file.write(model_json)
# Save weights to hdf5 file
model.save_weights("model_aug_weights.h5")
