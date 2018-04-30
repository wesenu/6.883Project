# Import
from keras.datasets import mnist
import matplotlib
import time

matplotlib.use('Agg')
# matplotlib inline
import matplotlib.pyplot as plt

import l2_attack
import keras
from defense import *




xin = tf.placeholder(tf.float32, [30, 128])

session = keras.backend.get_session()
mygan = Generator(30, xin)
print(mygan)

keras.backend.set_learning_phase(False)
model = keras.models.load_model("data/mnist")

touse = [x for x in tf.trainable_variables() if 'Generator' in x.name]
saver = tf.train.Saver(touse)
saver.restore(session, 'data/mnist-gan')
print("session",session)
time.sleep(6)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = np.array(x_test, dtype=np.float32)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test /= 255.0

# 1. solve the inversion problem
attack1 = l2_attack.CarliniL2(session, model,
                              lambda x: Generator(30, x),
                              binary_search_steps=1,
                              max_iterations=3000,
                              learning_rate=1e-1,
                              batch_size=30,
                              initial_const=0, targeted=None)

# 2. start there, make it adversarial
attack2 = l2_attack.CarliniL2(session, model,
                              lambda x: Generator(1, x),
                              binary_search_steps=5,
                              max_iterations=30000,
                              learning_rate=1e-1,
                              batch_size=1,
                              initial_const=1, targeted=False,
                              noise=False, abort_early=False)

#

#res1 = session.run(Generator(30, x))
#print(res1)

print(model)

print("True label", y_test[0])
print("Preds", model.predict(x_test)[0])
plt.imshow(x_test[0, :, :, 0])
# plt.show()
plt.savefig("figs/preds.png")


res = attack1.attack([x_test[0]] * 30,
                     [np.eye(10)[0]] * 30)

#print("shape",model.predict(res).shape)
#plt.figure()
#plt.imshow(res[0, :, :, 0])
#plt.savefig("figs/idea_wgan1.pdf")
#print("Preds", np.argmax(model.predict(res)[0]))


start = session.run(attack1.modifier)
it = session.run(mygan, {xin: start})
distortion = np.sum((it - [x_test[0]] * 30) ** 2, (1, 2, 3)) ** .5
print("Distortions", distortion)
start = np.array([start[np.argmin(distortion)]])

res = session.run(Generator(1, tf.constant(start, dtype=tf.float32)))
print("Preds", np.argmax(model.predict(res)))
plt.figure()
plt.imshow(res[0, :, :, 0])
plt.savefig("figs/idea_wgan1.pdf")
#print("!!res",res)
print(np.sum((res - x_test[:1]) ** 2, (1, 2, 3)) ** .5)
print(np.mean(np.sum((res - x_test[:1]) ** 2, (1, 2, 3)) ** .5))

plt.figure()
plt.imshow(res[0, :, :, 0])
#plt.show()
# #plt.savefig("figs/model3_wgan-gp.png")


res = attack2.attack(x_test[:1],
                     [np.eye(10)[q] for q in y_test[:1]],
                     start)


print("L2 Distortion", np.sum((res - x_test[:1]) ** 2) ** .5)

print(model.predict(res).shape)
print("Preds", np.argmax(model.predict(res)[0]))
plt.imshow(res[0, :, :, 0])
# plt.show()
plt.savefig("figs/3model_wgan1.pdf")







