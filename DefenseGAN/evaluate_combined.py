import numpy as np
import keras
import keras.backend as K

classifier_name = ['A', 'B', 'C']

for target in range(10):
    adv = np.load('../APEGAN/reconstruct/rec_t%d.npy' % target)
    #clean = np.load('../AdvGAN/samples/WB-B-t%d-clean.npy' % target)[10000:50000]
    data = np.load('../DefenseGAN/results/reconstruct/rec_t%d-adv-defensegan-5000.npz' % target)
    rec = data["arr_0"]
    print(rec.shape)
    #label = np.load('../APEGAN/reconstruct/label_%d.npy' % target)
    label = np.load('../AdvGAN/samples/WB-B-t%d-label.npy' % target)
    num = rec.shape[0]
    for cn in classifier_name:
        F = keras.models.load_model('../AdvGAN/models/Classifier-' + cn + '.h5')
        pdt_adv = np.argmax(F.predict(adv[:num,]), axis=1)
        pdt_rec = np.argmax(F.predict(rec[:num,]), axis=1)
        adv_acc = np.sum(pdt_adv==label[:num,])/num
        rec_acc = np.sum(pdt_rec==label[:num,])/num
        print('{}, {} : adv acc:{:.4f}, rec acc:{:.4f}'.format(target, cn, adv_acc, rec_acc))
