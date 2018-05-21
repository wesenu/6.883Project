import numpy as np
import keras
import keras.backend as K

classifier_name = ['A', 'B', 'C']

data = np.load('../cleverhans/adv-cw-modified-train-mnist-5000.npz')
adv = data["x"] 
#label = data["y"]
#clean = np.load('../AdvGAN/samples/WB-B-t%d-clean.npy' % target)[10000:50000]
data = np.load('results/reconstruct/cwm-defensegan-500.npz')
rec = data["arr_0"]
#label = np.load('../APEGAN/reconstruct/label_%d.npy' % target)
data = np.load('../cleverhans/1-adv-cw-orginal-train-mnist.npz')
label = data['y']
num = rec.shape[0]
print(num)
target = 0
for cn in classifier_name:
    F = keras.models.load_model('../AdvGAN/models/Classifier-' + cn + '.h5')
    pdt_adv = np.argmax(F.predict(adv[:num,]), axis=1)
    pdt_rec = np.argmax(F.predict(rec[:num,]), axis=1)
    adv_acc = np.sum(pdt_adv==label[:num,])/num
    rec_acc = np.sum(pdt_rec==label[:num,])/num
    print('{}, {} : adv acc:{:.4f}, rec acc:{:.4f}'.format(target, cn, adv_acc, rec_acc))

