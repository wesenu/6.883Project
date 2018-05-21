import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, ZeroPadding2D, Convolution2D, LeakyReLU, Add, Convolution2DTranspose, Conv2DTranspose, BatchNormalization, Lambda, Subtract, Dot, Concatenate
from keras_contrib.layers.normalization import InstanceNormalization
from keras.optimizers import Adam
from keras import backend as K


def MSE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def Adv(y_true, y_pred):
    target = K.sum(y_true * y_pred, axis=-1)
    other = K.max((1 - y_true) * y_pred, axis=-1)
    return K.maximum(other - target, 0)


def Hinge(c):
    def loss(y_true, y_pred):
        return  K.mean(K.maximum(K.square(y_true - y_pred) - c, 0))
    return loss


def SRMSE(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1) + 1e-10)


def Combined(input_shape, classifier_name):
    G = generator(input_shape)
    D = discriminator(input_shape, classifier_name)
    F = keras.models.load_model('./models/Classifier-' + classifier_name + '.h5')
    ipt = Input(input_shape)
    perturbation = G(ipt)
    adversary = Add()([ipt, perturbation])
    D.trainable = False
    F.trainable = False
    judge = D(adversary)[0]
    scores = F(adversary)

    GAN = Model(ipt, [judge, scores, perturbation])
    GAN.compile(optimizer='adam',
                loss=[MSE, Adv, Hinge(0.005)],
                loss_weights=[0.5, 2, 1.5])
    return GAN, G, D, F


def Adv_generator(input_shape=[28,28,1]):
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
    return M


def APE_generator(input_shape):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=2, padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, (3,3), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(64, (3,3), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv2DTranspose(1, (3,3), strides=2, padding='same'))
    model.add(Activation('tanh'))
    return model


def generator(input_shape):
    return Adv_generator(input_shape)


def discriminator(input_shape, classifier_name):
    F = keras.models.load_model('./models/Classifier-' + classifier_name + '.h5')
    F.trainable = False
    APEG = APE_generator(input_shape)
    m_in = Input(shape=input_shape)
    m_adv_pdt = F(m_in)
    m_rct = APEG(m_in)
    m_rct_pdt = F(m_rct)
    m_pdt_delta = Concatenate(axis=1)([m_rct_pdt, m_adv_pdt])
    m_pdt_delta = Dense(20)(LeakyReLU(0.2)(m_pdt_delta))
    m_pdt_delta = Dense(20)(LeakyReLU(0.2)(m_pdt_delta))
    m_out = Activation('sigmoid')(Dense(1)(LeakyReLU(0.2)(m_pdt_delta)))
    #m_out = Activation('tanh')(Dot(axes=1)([m_pdt_diff, m_pdt_diff]))
    M = Model(m_in, [m_out, m_rct])
    M.compile(optimizer='adam', loss=[MSE, SRMSE], loss_weights=[0.3, 0.7])
    return M