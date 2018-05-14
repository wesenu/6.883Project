from Generator import Define_Generator
from Discriminator import Define_Discriminator
from Distilled import Define_Distilled

from keras.layers import Input, Add
from keras.models import Model
from keras.optimizers import adam

from Losses import Custom_MSE, Adv, Custom_Hinge


def Define_GAN(input_shape, alpha, beta):
    G = Define_Generator(input_shape)
    D = Define_Discriminator(input_shape)
    f = Define_Distilled(input_shape)

    x_inp = Input(input_shape)
    perturb = G(x_inp)
    x_perturbed = Add()([x_inp, perturb])

    Discrim_Output = D(x_perturbed)
    Class_Output = f(x_perturbed)

    GAN = Model(x_inp, [Discrim_Output, Class_Output, perturb])
    GAN.compile(optimizer=adam(lr=0.001),
                loss=[Custom_MSE, Adv, Custom_Hinge(0.3)],
                loss_weights=[1, alpha, beta])
    GAN.summary()

    return GAN, G, D, f
