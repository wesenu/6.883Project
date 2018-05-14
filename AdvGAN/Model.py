import tensorflow as tf
import keras
from keras.models import Model, Sequential  # basic class for specifying and training a neural network
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Convolution2D, ZeroPadding2D, Activation, Add, Convolution2DTranspose
from keras_contrib.layers.normalization import InstanceNormalization
from keras.layers import Input, ZeroPadding2D, Convolution2D, LeakyReLU, Flatten, Dense
from keras_contrib.layers.normalization import InstanceNormalization
from keras.models import Model
from Losses import MSE, Adv, Hinge
from keras.layers import Input, Add
from keras.models import Model
from keras.optimizers import adam
import keras


def GAN(input_shape, classifier_name, alpha, beta):
    G = generator(input_shape)
    D = discriminator(input_shape)
    F = keras.models.load_model('./models/Classifier-' + classifier_name + '.h5')
    ipt = Input(input_shape)
    perturbation = G(ipt)
    adversary = Add()([ipt, perturbation])
    D.trainable = False
    F.trainable = False
    judge = D(adversary)
    scores = F(adversary)

    GAN = Model(ipt, [judge, scores, perturbation])
    GAN.compile(optimizer=adam(lr=0.001),
                loss=[MSE, Adv, Hinge(0.1)],
                loss_weights=[1, alpha, beta])
    return GAN, G, D, F


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


def generator(input_shape=[28,28,1]):
    def conv__inst_norm__relu(x_input, filters, kernel_size=(3, 3), stride=1):
        l = ZeroPadding2D()(x_input)
        l = Convolution2D(filters=filters, kernel_size=(3, 3), strides=stride, activation='linear')(l)
        l = InstanceNormalization()(l)
        l = Activation('relu')(l)
        return l

    def res__block(x_input, filters, kernel_size=(3, 3), stride=1):
        l = ZeroPadding2D()(x_input)
        l = Convolution2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=stride, )(l)
        l = InstanceNormalization()(l)
        l = Activation('relu')(l)

        l = ZeroPadding2D()(l)
        l = Convolution2D(filters=filters,
                          kernel_size=kernel_size,
                          strides=stride, )(l)
        l = InstanceNormalization()(l)
        merged = Add()([x_input, l])
        return merged

    def trans_conv__inst_norm__relu(x_input, filters, kernel_size=(3, 3), stride=2):
        l = Convolution2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride, activation='linear',
                                   padding='same')(x_input)
        l = InstanceNormalization()(l)
        l = Activation('relu')(l)
        return l

    m_in = Input(shape=input_shape)
    m = conv__inst_norm__relu(m_in, filters=8, stride=1)
    m = conv__inst_norm__relu(m, filters=16, stride=2)
    m = conv__inst_norm__relu(m, filters=32, stride=2)
    m = res__block(m, filters=32)
    m = res__block(m, filters=32)
    m = res__block(m, filters=32)
    m = res__block(m, filters=32)
    m = trans_conv__inst_norm__relu(m, filters=16, stride=2)
    m = trans_conv__inst_norm__relu(m, filters=8, stride=2)
    m_out = conv__inst_norm__relu(m, filters=1, stride=1)
    M = Model(m_in, m_out)
    M.compile(optimizer='adam', loss='mean_squared_error')
    return M


def discriminator(input_shape=[28, 28, 1]):
    m_in = Input(shape=input_shape)
    m = ZeroPadding2D()(m_in)
    m = Convolution2D(filters=8,
                      kernel_size=(4, 4),
                      strides=2)(m)
    m = InstanceNormalization()(m)
    m = LeakyReLU(0.2)(m)
    m = ZeroPadding2D()(m)
    m = Convolution2D(filters=16,
                      kernel_size=(4, 4),
                      strides=2)(m)
    m = InstanceNormalization()(m)
    m = LeakyReLU(0.2)(m)
    m = ZeroPadding2D()(m)
    m = Convolution2D(filters=32,
                      kernel_size=(4, 4),
                      strides=2)(m)
    m = InstanceNormalization()(m)
    m = LeakyReLU(0.2)(m)
    m = Flatten()(m)
    m_out = Dense(1, activation='sigmoid')(m)
    M = Model(m_in, m_out)
    M.compile(optimizer='adam', loss='mean_squared_error')
    return M


