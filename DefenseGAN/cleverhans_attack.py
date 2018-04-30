# Import
from keras.datasets import mnist
import matplotlib
import time

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

matplotlib.use('Agg')
# matplotlib inline
import matplotlib.pyplot as plt

import l2_attack
import keras
from defense import *

# cleverhans

from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval, tf_model_load

# from cleverhans_tutorials.tutorial_models import make_basic_cnn

xin = tf.placeholder(tf.float32, [30, 128])

session = keras.backend.get_session()
mygan = Generator(30, xin)

keras.backend.set_learning_phase(False)
model = keras.models.load_model("data/mnist")
print("model", model)
# print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file("data/mnist", tensor_name='', all_tensors=True)

touse = [x for x in tf.trainable_variables() if 'Generator' in x.name]
saver = tf.train.Saver(touse)
saver.restore(session, 'data/mnist-gan')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.array(x_test, dtype=np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test /= 255.0
print(x_test[:10].shape)

train_start = 0
train_end = 60000
test_start = 0
test_end = 10000

X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                              train_end=train_end,
                                              test_start=test_start,
                                              test_end=test_end)

# 1. solve the inversion problem
attack1 = l2_attack.CarliniL2(session, model,
                              lambda x: Generator(30, x),
                              binary_search_steps=1,
                              max_iterations=3000,
                              learning_rate=1e-1,
                              batch_size=30,
                              initial_const=0, targeted=None)

test_start = 0
test_end = 10000

channels = 1
nb_classes = 100
batch_size = 128

x = tf.placeholder(tf.float32, shape=(None, 28, 28, channels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))
preds = model(x)
print(session.run(preds, {x: x_test[:10]}))
# print(model.predict(x_test[:10]))
# print(y_test[:10])
# print(np.argmax(pred_result))
# preds = model.predict(x)
# pred_result = session.run(preds,{x: x_test[:10]})


# cw = CarliniWagnerL2(model, back='tf', sess=session)

# Evaluate the accuracy of the MNIST model on legitimate test examples
#
eval_params = {'batch_size': batch_size}
accuracy = model_eval(session, x, y, preds, X_test, Y_test, args=eval_params)
assert x_test.shape[0] == test_end - test_start, x_test.shape
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

source_samples = 10
targeted = True
nb_adv_per_sample = str(nb_classes - 1) if targeted else '1'
print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
      ' adversarial examples')
print("This could take some time ...")

cw = CarliniWagnerL2(model, back='tf', sess=session)

adv_inputs = X_test[:source_samples]

yname = "y"
adv_ys = None
attack_iterations = 100


cw_params = {'binary_search_steps': 1,
             yname: adv_ys,
             'max_iterations': attack_iterations,
             'learning_rate': 0.1,
             'batch_size': source_samples * nb_classes if
             targeted else source_samples,
             'initial_const': 100}

adv = cw.generate_np(adv_inputs, **cw_params)

adv_accuracy = 1 - model_eval(session, x, y, preds, adv, Y_test[
                                                      :source_samples], args=eval_params)
print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))

# print("True label", y_test[0])
# print("Preds", model.predict(x_test)[0])
# plt.imshow(x_test[0, :, :, 0])
# plt.show()
# plt.savefig("figs/preds.png")

# res = attack1.attack([x_test[0]] * 30,[np.eye(10)[0]] * 30)

# start = session.run(attack1.modifier)

# it = session.run(mygan, {xin: start})
# distortion = np.sum((it - [x_test[0]] * 30) ** 2, (1, 2, 3)) ** .5
# print("Distortions", distortion)
# start = np.array([start[np.argmin(distortion)]])

# res = session.run(Generator(1, tf.constant(start, dtype=tf.float32)))
# print("!!res",res)
# print(np.sum((res - x_test[:1]) ** 2, (1, 2, 3)) ** .5)
# print(np.mean(np.sum((res - x_test[:1]) ** 2, (1, 2, 3)) ** .5))

# res = attack2.attack(x_test[:1],[np.eye(10)[q] for q in y_test[:1]],start)

# print("L2 Distortion", np.sum((res - x_test[:1]) ** 2) ** .5)

# print("Preds", model.predict(res)[0])
# plt.imshow(res[0, :, :, 0])
# plt.show()
# plt.savefig("figs/model.png")
# res = session.run(Generator(1, tf.constant(start, dtype=tf.float32)))
# print("!!res",res)
# print("res shape",res.shape)
