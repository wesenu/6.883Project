import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, ZeroPadding2D, Activation, Add, Convolution2DTranspose
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, ZeroPadding2D, Convolution2D, LeakyReLU, Flatten, Dense, Add
from keras.optimizers import Adam
from keras import backend as K


def classifier(model_name):
    model = None
    if model_name == 'A':
        model = Sequential()
        model.add(Conv2D(64, (5, 5), padding='valid',
                         input_shape=(28,
                                      28,
                                      1)))
        model.add(Activation('relu'))

        model.add(Conv2D(64, (5, 5), padding='valid'))
        model.add(Activation('relu'))

        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        opt = keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    if model_name == 'B':
        model = Sequential()
        model.add(Conv2D(64, (8, 8), padding='valid',
                         input_shape=(28,
                                      28, 1)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Conv2D(128, (6, 6), padding='valid'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (5, 5), padding='valid'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(10))

        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

    if model_name == 'C':
        params = [32, 32, 64, 64, 200, 200]

        model = Sequential()

        model.add(Conv2D(params[0], (3, 3),
                         input_shape=(28,
                                      28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(params[1], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(params[2], (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(params[3], (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(params[4]))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(params[5]))
        model.add(Activation('relu'))
        model.add(Dense(10, activation='softmax'))

        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
