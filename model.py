from keras.layers import Conv2D, AveragePooling2D, GlobalAveragePooling2D, Dense
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers import Input, Concatenate
from keras.regularizers import l2
from keras.models import Model


def conv_layer(x, concat_axis, num_filters, dropout=None, weight_decay=1e-4):
    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=4 * num_filters, kernel_size=(1, 1), strides=(1, 1), kernel_initializer="he_uniform",
               use_bias=False, kernel_regularizer=l2(weight_decay), padding="same")(x)
    if dropout:
        x = Dropout(dropout)(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1), kernel_initializer="he_uniform",
               use_bias=False, kernel_regularizer=l2(weight_decay), padding="same")(x)
    if dropout:
        x = Dropout(dropout)(x)

    return x


def transition_layer(x, concat_axis, num_filters, dropout=None, weight_decay=1e-4):

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = Conv2D(filters=num_filters, kernel_size=(1, 1), strides=(1, 1), kernel_initializer="he_uniform",
               use_bias=False, kernel_regularizer=l2(weight_decay), padding="same")(x)
    if dropout:
        x = Dropout(dropout)(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid")(x)

    return x


def dense_block(x, concat_axis, num_layers, num_filters, growth_rate, dropout=None, weight_decay=1e-4):

    feat_maps = [x]
    for i in range(num_layers):
        x = conv_layer(x=x, concat_axis=concat_axis, num_filters=growth_rate,
                       dropout=dropout, weight_decay=weight_decay)
        feat_maps.append(x)
        x = Concatenate(axis=concat_axis)(feat_maps)
        num_filters += growth_rate

    return x, num_filters


def dense_net(input_shape, depth, num_dense_block, num_filters, growth_rate, dropout_rate=None,
              concate_axis=-1, weight_decay=1e-4):

    input_x = Input(shape=input_shape)

    num_layers = int((depth - 4) / 3)

    x = Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(1, 1),
               kernel_initializer="he_uniform", padding="same", use_bias=False,
               kernel_regularizer=l2(weight_decay), name="initial_conv")(input_x)

    for i in range(num_dense_block - 1):
        x, num_filters = dense_block(x=x, concat_axis=concate_axis, num_layers=num_layers,
                                     num_filters=num_filters, growth_rate=growth_rate,
                                     dropout=dropout_rate, weight_decay=weight_decay)
        x = transition_layer(x=x, concat_axis=concate_axis, num_filters=num_filters,
                             dropout=dropout_rate, weight_decay=weight_decay)

    x, num_filters = dense_block(x=x, concat_axis=concate_axis, num_layers=num_layers,
                                 num_filters=num_filters, growth_rate=growth_rate,
                                 dropout=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(axis=concate_axis, gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D(data_format="channels_last")(x)
    x = Dense(units=100, activation="softmax", kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    network_model = Model(inputs=[input_x], outputs=[x], name="DenseNet")

    network_model.summary()

    return network_model


if __name__ == "__main__":
    model = dense_net(input_shape=(24, 24, 1), depth=40, num_dense_block=3, num_filters=16, growth_rate=12,
                      dropout_rate=0.2, concate_axis=-1, weight_decay=1e-4)
