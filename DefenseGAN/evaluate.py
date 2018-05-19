import keras
import keras.backend as K
import numpy as np

classifier_name = ['A', 'B', 'C']

WB_B_t1_adv = np.load('../AdvGAN/samples/WB-B-t1-adv.npy')
WB_B_t1_clean = np.load('../AdvGAN/samples/WB-B-t1-clean.npy')
WB_B_t1_label = np.load('../AdvGAN/samples/WB-B-t1-label.npy')
print(WB_B_t1_label.shape)

#WB_B_t0_defensegan = np.load('results/reconstruct-1000/WB-B-t0-adv-defensegan-1000.npz')['arr_0']
WB_B_t1_defensegan = np.load('results/reconstruct/WB-B-t1-adv-defensegan-51596.npz')['arr_0']


#print(WB_B_t0_label[img])



for cn in classifier_name:
    #F = keras.models.load_model('../AdvGAN/models/Classifier-' + cn + '.h5')
    F = keras.models.load_model('../AdvGAN/models/Classifier-A.h5')
    length = WB_B_t1_defensegan.shape[0]
    score = F.evaluate(WB_B_t1_defensegan, WB_B_t1_label[:length], verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
