from keras import backend as K


def MSE(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def Adv(y_true, y_pred):
    target = K.sum(y_true * y_pred, axis=-1)
    other = K.max((1 - y_true) * y_pred, axis=-1)
    return K.maximum(other - target, 0)


def Hinge(c):
    def loss(y_true, y_pred):
        return  K.maximum(MSE(y_true, y_pred) - c, 0)
    return loss