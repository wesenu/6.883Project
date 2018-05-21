import numpy as np
clean=np.load('WB-B-t0-clean.npy')
adv=np.load('WB-B-t0-adv.npy')
label=np.load('WB-B-t0-label.npy')
train_clean=clean[:10000]
train_adv=adv[:10000]
test_adv=adv[10000:50000]
test_lab=label[10000:50000]

np.save('clean_0.npy',train_clean)
np.save('adv_0.npy',train_adv)
np.save('label_0.npy',test_lab)
np.save('test_0.npy',test_adv)

clean=np.load('WB-B-t5-clean.npy')
adv=np.load('WB-B-t5-adv.npy')
label=np.load('WB-B-t5-label.npy')
train_clean=clean[:10000]
train_adv=adv[:10000]
test_adv=adv[10000:50000]
test_lab=label[10000:50000]

np.save('clean_5.npy',train_clean)
np.save('adv_5.npy',train_adv)
np.save('label_5.npy',test_lab)
np.save('test_5.npy',test_adv)
